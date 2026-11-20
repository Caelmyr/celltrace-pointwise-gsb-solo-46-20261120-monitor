import {
  $, $$, badge, bindCanvas, button, e, field, normalizedPoint,
  request, submit, table, wireForm,
} from "./ui.js";

const SEAM_COLOR = "#f5c469";
const NOMINAL_COLOR = "#6cadab";
const SAMPLE_COLORS = {
  aligned: "#8be7c9", offset: "#f5bc5e", "weak separator": "#e98aa3",
  "ambiguous valley": "#e98aa3", "multiple comparable valleys": "#e98aa3",
  "low shoulders": "#e98aa3", unsupported: "#9a8fa8",
};

export function showSeamCalibration(app, role) {
  const root = app.openDialog("Calibrate cell seam", '<p class="small-copy">Loading separator review…</p>', true);
  request(app.url("/seam-review", {role})).then((data) => drawDialog(app, root, role, data, null)).catch((error) => {
    if (root.isConnected) root.innerHTML = `<p class="error">${e(error.message)}</p>`;
  });
}

function drawDialog(app, root, role, data, conflicts) {
  const disabled = app.case.state === "issued";
  const seams = [...data.vertical, ...data.horizontal];
  const calibratedCount = seams.filter((seam) => seam.calibrated).length;
  root.innerHTML = `<form class="section-stack">
    <p class="small-copy">Pick the seam that drifts, then click the dark separator line on the
      rectified panel to place calibration points. The two neighbouring cells share the corrected
      boundary, so no strip is left unmeasured or counted twice. Cell IDs (RxxCxx) never change.</p>
    <div class="fields">
      ${field("Seam", "seam", "", {values: seams.map((seam) => [seamKey(seam), seamLabel(seam)]), wide: true})}
    </div>
    <div class="dialog-view"><div class="viewer-viewport">
      <img id="seam-image" alt="Rectified panel with separator samples"
        src="${e(app.url("/layer/panel.png", {role, grid: 0, labels: 0, revision: app.case.revision}))}">
      <canvas id="seam-canvas" tabindex="0" aria-label="Click the dark separator to add a calibration point"></canvas>
    </div></div>
    <div id="seam-editor"></div>
    ${conflicts && conflicts.length ? conflictPanel(conflicts) : ""}
    <p class="form-error" hidden></p>
    <div class="form-actions">
      ${button("seam-reset-current", "Reset this seam", "", disabled)}
      ${disabled ? "" : submit("Save seam calibration")}
    </div>
    <p class="small-copy">Calibrated seams: ${calibratedCount} / ${seams.length}. Points use rectified plane pixels.</p>
  </form>`;
  const selector = $("select[name=seam]", root);
  let current = seams.find((seam) => seamKey(seam) === selector.value) || seams[0];
  let points = current.calibrated ? current.calibrated.points.map(([a, b]) => [a, b]) : [];
  const image = $("#seam-image", root);
  const canvas = $("#seam-canvas", root);

  function redraw() {
    bindCanvas(canvas, image, (painter, width, height) => {
      paintSeam(painter, width, height, data, current, points, app.case.plane);
    });
    renderEditor(root, data, current, points, disabled);
  }
  selector.onchange = () => {
    current = seams.find((seam) => seamKey(seam) === selector.value);
    points = current.calibrated ? current.calibrated.points.map(([a, b]) => [a, b]) : [];
    redraw();
  };
  canvas.onclick = (event) => {
    if (disabled) return;
    const [nx, ny] = normalizedPoint(event, canvas);
    const plane = app.case.plane;
    const along = current.axis === "vertical" ? ny * plane.height : nx * plane.width;
    const across = current.axis === "vertical" ? nx * plane.width : ny * plane.height;
    if (points.length >= 24) return app.notify("A seam supports up to 24 calibration points.", true);
    points.push([round1(along), round1(across)]);
    points.sort((a, b) => a[0] - b[0]);
    redraw();
  };
  $("#seam-reset-current", root).onclick = async (event) => {
    event.preventDefault();
    try {
      const saved = await request(app.url("/grid-seam-reset"), "PATCH", {
        revision: app.case.revision, axis: current.axis, index: current.index,
      });
      await app.update(saved, "Seam returned to the regular grid.");
      const fresh = await request(app.url("/seam-review", {role}));
      drawDialog(app, root, role, fresh, null);
    } catch (error) {
      app.fail(error);
    }
  };
  wireForm(root, async () => {
    if (points.length < 2) throw new Error("Place at least two calibration points along the seam.");
    const saved = await request(app.url("/grid-seam"), "PATCH", {
      revision: app.case.revision, axis: current.axis, index: current.index, points,
    });
    const nextConflicts = saved.seam_conflicts || [];
    await app.update(saved, nextConflicts.length
      ? "Seam saved. Review the annotation ownership reminders."
      : "Seam calibration saved.");
    const fresh = await request(app.url("/seam-review", {role}));
    drawDialog(app, root, role, fresh, nextConflicts);
  });
  redraw();
}

function seamKey(seam) {
  return `${seam.axis}-${seam.index}`;
}

function seamLabel(seam) {
  return seam.calibrated ? `${seam.label} · calibrated` : seam.label;
}

function round1(value) {
  return Math.round(value * 10) / 10;
}

function paintSeam(painter, width, height, data, current, points, plane) {
  const vertical = current.axis === "vertical";
  // Nominal straight boundary for every seam of the same orientation.
  painter.lineWidth = 1;
  for (const seam of data[vertical ? "vertical" : "horizontal"]) {
    painter.strokeStyle = seamKey(seam) === seamKey(current) ? NOMINAL_COLOR : "rgba(108,173,171,0.45)";
    drawNominal(painter, width, height, seam, plane);
  }
  // Detected separator samples from the original image.
  for (const sample of current.samples) {
    painter.fillStyle = SAMPLE_COLORS[sample.status] || "#e98aa3";
    const [span0, span1] = sample.span;
    if (vertical) {
      const x = (sample.expected / plane.width) * width;
      const y0 = (span0 / plane.height) * height;
      const y1 = (span1 / plane.height) * height;
      painter.fillRect(x - 2, y0, 4, Math.max(2, y1 - y0));
    } else {
      const y = (sample.expected / plane.height) * height;
      const x0 = (span0 / plane.width) * width;
      const x1 = (span1 / plane.width) * width;
      painter.fillRect(x0, y - 2, Math.max(2, x1 - x0), 4);
    }
  }
  // Calibrated piecewise seam from saved points, plus the points being edited.
  if (points.length) {
    painter.strokeStyle = SEAM_COLOR;
    painter.lineWidth = 2;
    painter.beginPath();
    points.forEach(([along, across], index) => {
      const x = (vertical ? across : along) / plane.width * width;
      const y = (vertical ? along : across) / plane.height * height;
      if (index === 0) painter.moveTo(x, y);
      else painter.lineTo(x, y);
    });
    painter.stroke();
    for (const [along, across] of points) {
      const x = (vertical ? across : along) / plane.width * width;
      const y = (vertical ? along : across) / plane.height * height;
      painter.beginPath();
      painter.arc(x, y, 4.5, 0, Math.PI * 2);
      painter.fillStyle = SEAM_COLOR;
      painter.fill();
    }
  }
}

function drawNominal(painter, width, height, seam, plane) {
  painter.beginPath();
  if (seam.axis === "vertical") {
    const x = seam.index / plane.cols * width;
    painter.moveTo(x, 0);
    painter.lineTo(x, height);
  } else {
    const y = seam.index / plane.rows * height;
    painter.moveTo(0, y);
    painter.lineTo(width, y);
  }
  painter.stroke();
}

function renderEditor(root, data, seam, points, disabled) {
  const editor = $("#seam-editor", root);
  const plane = data.plane ? {width: data.plane[0], height: data.plane[1]} : null;
  const alongLabel = seam.axis === "vertical" ? "Y position" : "X position";
  const acrossLabel = seam.axis === "vertical" ? "Seam X" : "Seam Y";
  const rows = seam.samples.map((sample) => ({
    band: `${sample.span[0]}–${sample.span[1]}`,
    expected: sample.expected,
    detected: sample.detected ?? "—",
    offset: sample.offset ?? "—",
    status: sample.status,
  }));
  editor.innerHTML = `<div class="section-stack">
    <div class="toolbar"><span class="chip">${e(seam.label)}</span>
      ${badge(`${points.length} point${points.length === 1 ? "" : "s"}`, "neutral")}</div>
    ${points.length ? `<div class="table-wrap"><table><thead><tr>
      <th>${e(alongLabel)}</th><th>${e(acrossLabel)}</th><th></th></tr></thead><tbody>
      ${points.map((point, index) => `<tr><td class="num">${e(point[0])}</td><td class="num">${e(point[1])}</td>
        <td><button type="button" class="compact" data-remove="${index}" ${disabled ? "disabled" : ""}>Remove</button></td></tr>`).join("")}
      </tbody></table></div>` : '<p class="small-copy">Click the dark separator on the image to add the first two points.</p>'}
    <details><summary>Separator samples from the original image</summary>
      ${table([["Band px", "band"], ["Declared", "expected"], ["Detected", "detected"], ["Offset", "offset"], ["Status", "status"]], rows)}
    </details></div>`;
  $$("[data-remove]", editor).forEach((node) => {
    node.onclick = () => {
      points.splice(Number(node.dataset.remove), 1);
      renderEditor(root, data, seam, points, disabled);
      bindCanvas($("#seam-canvas", root), $("#seam-image", root), (painter, width, height) => {
        paintSeam(painter, width, height, data, seam, points,
          {width: data.plane[0], height: data.plane[1], rows: data.grid[0], cols: data.grid[1]});
      });
    };
  });
}

function conflictPanel(conflicts) {
  return `<section class="conflict-panel" aria-label="Annotation ownership review">
    <h3>Review annotation ownership (${conflicts.length})</h3>
    <p class="small-copy">The corrected seam passes through these saved annotations. Their original
      position and recorded cell ID are preserved; confirm which cell each belongs to now.</p>
    <div class="table-wrap"><table><thead><tr><th>Finding</th><th>Recorded cell</th><th>Now spans</th><th>Evidence</th></tr></thead>
      <tbody>${conflicts.map((item) => `<tr>
        <td>${e(item.title)}<br><small>${e(item.finding)}</small></td>
        <td>${e(item.recorded_cell || "—")}</td>
        <td>${e(item.current_cells.join(", "))}</td>
        <td>${badge(item.source_status === "current" ? "Current source" : "Source changed",
          item.source_status === "current" ? "neutral" : "training")}</td>
      </tr>`).join("")}</tbody></table></div></section>`;
}

export function seamConflictBanner(conflicts) {
  if (!conflicts || !conflicts.length) return "";
  return `<section class="card warning-banner"><div class="card-pad">
    <strong>${conflicts.length} annotation${conflicts.length === 1 ? "" : "s"} need an ownership review.</strong>
    <p class="small-copy">A calibrated seam now crosses these findings. Their saved position and cell ID were kept;
      open the Findings tab to confirm ownership.</p>
    <div class="table-wrap"><table><tbody>${conflicts.map((item) => `<tr>
      <td>${e(item.title)}</td><td>${e(item.recorded_cell || "—")}</td>
      <td>${e(item.current_cells.join(", "))}</td></tr>`).join("")}</tbody></table></div>
  </div></section>`;
}
