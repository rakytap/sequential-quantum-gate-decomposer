const byId = id => document.getElementById(id);
const gateSelect = byId("gate");
const basisSelect = byId("basis");
const exact = byId("exact");
const format = byId("format");
const fields = byId("parameter-fields");
const basisSearch = byId("basis-search");
const copy = byId("copy");
const download = byId("download");
let data;
let result;
let qasm = "";

function element(tag, content, className) {
  const item = document.createElement(tag);
  if (content !== undefined) item.textContent = content;
  if (className) item.className = className;
  return item;
}

function options(select, values) {
  select.replaceChildren(...values.map(value => new Option(value, value)));
}

function svgElement(parent, tag, attributes, content) {
  const item = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attributes)) item.setAttribute(key, value);
  if (content !== undefined) item.textContent = content;
  parent.append(item);
  return item;
}

function operationText([gate, , parameters]) {
  return gate + (parameters.length
    ? "(" + parameters.map(index => data.expressions[index][0]).join(", ") + ")"
    : "");
}

function drawCircuit(width, operations, entangler) {
  const container = byId("diagram");
  container.replaceChildren();
  const svg = svgElement(container, "svg", {role: "img", "aria-labelledby": "diagram-title"});
  svgElement(svg, "title", {id: "diagram-title"}, width + "-qubit circuit with " + operations.length + " operations");
  const wireY = wire => 30 + wire * 56;
  let position = 64;
  const placements = operations.map(operation => {
    const [gate, wires, parameters] = operation;
    const args = parameters.map(index => data.expressions[index][0]).join(", ");
    const label = args.length < 16 ? args : "…";
    const boxWidth = gate === "CX" || gate === "CZ" ? 28 :
      Math.max(40, gate.length * 8 + 20, label.length * 6 + 14);
    const x = position + boxWidth / 2;
    position += boxWidth + 14;
    return {operation, gate, wires, label, x, boxWidth};
  });
  const plotWidth = Math.max(280, position + 12);
  const plotHeight = width ? wireY(width - 1) + 32 : 64;
  svg.setAttribute("viewBox", "0 0 " + plotWidth + " " + plotHeight);
  svg.setAttribute("width", plotWidth);
  svg.setAttribute("height", plotHeight);
  for (let wire = 0; wire < width; wire++) {
    const y = wireY(wire);
    svgElement(svg, "text", {x: 14, y, class: "wire-label"}, "q[" + wire + "]");
    svgElement(svg, "line", {x1: 57, x2: plotWidth - 14, y1: y, y2: y, class: "wire"});
  }
  for (const {operation, gate, wires, label, x, boxWidth} of placements) {
    const group = svgElement(svg, "g", {class: gate === entangler ? "entangler" : "local"});
    svgElement(group, "title", {}, operationText(operation) + " on " + wires.map(wire => "q[" + wire + "]").join(", "));
    if (wires.length > 1) {
      svgElement(group, "line", {x1: x, x2: x, y1: wireY(Math.min(...wires)), y2: wireY(Math.max(...wires)), class: "coupler"});
    }
    wires.forEach((wire, operand) => {
      const y = wireY(wire);
      if (gate === "CX" || gate === "CZ") {
        if (gate === "CX" && operand === 1) {
          svgElement(group, "circle", {cx: x, cy: y, r: 12, class: "cx-target"});
          svgElement(group, "path", {d: "M " + (x - 8) + " " + y + " h 16 M " + x + " " + (y - 8) + " v 16", class: "cx-plus"});
        } else {
          svgElement(group, "circle", {cx: x, cy: y, r: 5.5, class: "control"});
        }
        return;
      }
      const height = label ? 38 : 29;
      svgElement(group, "rect", {x: x - boxWidth / 2, y: y - height / 2, width: boxWidth, height, rx: 5, class: "box"});
      svgElement(group, "text", {x, y: y - (label ? 7 : 0), class: "gate-label"}, gate);
      if (label) svgElement(group, "text", {x, y: y + 9, class: "angle-label"}, label);
      if (wires.length > 1) {
        svgElement(group, "text", {x: x + boxWidth / 2 + 4, y: y - 9, class: "operand-label"}, String(operand + 1));
      }
    });
  }
  if (!operations.length) {
    svgElement(svg, "rect", {x: 100, y: 16, width: 125, height: 28, rx: 5, class: "identity"});
    svgElement(svg, "text", {x: 162, y: 30, class: "gate-label"}, "Identity circuit");
  }
  container.scrollLeft = 0;
}

function drawOperations(operations, bindings) {
  const table = element("table");
  const head = table.createTHead().insertRow();
  for (const name of ["Step", "Operation", "Operands in order"]) head.append(element("th", name));
  const body = table.createTBody();
  operations.forEach((operation, index) => {
    const row = body.insertRow();
    row.insertCell().textContent = index + 1;
    row.insertCell().append(element("code", operationText(operation)));
    row.insertCell().textContent = operation[1].map(wire => "q[" + wire + "]").join(", ");
  });
  byId("operation-table").replaceChildren(table);
  const formulas = byId("formula-details");
  formulas.hidden = !bindings;
  formulas.open = false;
  byId("formula-bindings").textContent = (bindings || []).map(([name, index]) =>
    name + " = " + data.expressions[index][0]).join("\n");
}

function drawBases() {
  const query = basisSearch.value.trim().toLowerCase();
  const classes = {
    exact_continuous: "Exact continuous",
    approximate_discrete: "Approximate discrete",
    clifford: "Clifford only",
  };
  const matches = data.bases.filter(basis =>
    [basis.name, basis.entangler, ...basis.locals, classes[basis.classification], basis.notes]
      .join(" ").toLowerCase().includes(query));
  const body = byId("basis-table");
  body.replaceChildren();
  for (const basis of matches) {
    const row = body.insertRow();
    const button = element("button", basis.name, "basis-link");
    button.type = "button";
    if (basis.name === basisSelect.value) button.setAttribute("aria-current", "true");
    button.addEventListener("click", () => {
      basisSelect.value = basis.name;
      showSelection();
      gateSelect.focus();
      byId("top").scrollIntoView();
    });
    row.insertCell().append(button);
    row.insertCell().append(element("code", basis.entangler));
    row.insertCell().append(element("code", basis.locals.join(", ")));
    row.insertCell().textContent = classes[basis.classification];
    row.insertCell().textContent = basis.notes || "—";
  }
  if (!matches.length) {
    const cell = body.insertRow().insertCell();
    cell.colSpan = 5;
    cell.textContent = "No bases match your filter.";
  }
  byId("basis-count").textContent = query
    ? matches.length + " of " + data.bases.length + " native bases"
    : data.bases.length + " native bases";
}

function numberLabel(count, singular, plural) {
  return count + " " + (count === 1 ? singular : plural);
}

function drawQasm() {
  qasm = "";
  byId("copy-status").textContent = "";
  copy.textContent = "Copy";
  if (!result) {
    byId("qasm").textContent = "";
    return;
  }
  const {gate, basis, mode, target, canExportSymbolic} = result;
  const bound = format.value === "bound";
  const note = byId("export-note");
  fields.hidden = !bound;
  note.textContent = bound
    ? (canExportSymbolic ? "Exports a numerical instance at these values."
        : "The archived formulas use expressions that this exporter evaluates before writing QASM.")
    : target.inputs.length ? "Reusable gate with an example call in the export." : "";
  if (bound && target.inputs.length && target.convention) {
    note.append(" ", target.convention.replace(/\.?$/, "."));
  }
  try {
    const values = bound ? Array.from(fields.querySelectorAll("input"), input => {
      const value = input.valueAsNumber;
      if (!Number.isFinite(value)) throw Error("Enter a finite value for " + input.name + ".");
      return [input.name, value];
    }) : [];
    qasm = exportQasm(gate, basis, mode, target, data.expressions, format.value, values);
    byId("qasm").textContent = qasm;
  } catch (error) {
    byId("qasm").textContent = error.message;
  }
  copy.disabled = download.disabled = !qasm;
}

function showSelection() {
  const gate = gateSelect.value;
  const basis = basisSelect.value;
  const target = data.targets[gate];
  const native = data.bases.find(candidate => candidate.name === basis);
  const [exactMode, projectiveMode] = data.rows[gate][basis];
  const reusesExact = !exact.checked && projectiveMode === null;
  const mode = exact.checked || reusesExact ? exactMode : projectiveMode;
  const verified = mode.status === "verified_symbolic";
  const status = byId("status");
  const inputs = target.inputs;
  const signature = inputs.length ? "(" + inputs.join(", ") + ")" : "";
  byId("result-title").textContent = gate + signature;
  byId("result-basis").textContent = basis + " · " + native.entangler + " + " + native.locals.join(", ") +
    (target.scope === "register_fixture" ? " · Atlas fixture" : "");
  byId("verified").hidden = !verified;
  const counts = byId("counts");
  counts.replaceChildren();
  result = null;
  if (!verified) {
    const label = mode.status === "proved_impossible" ? "Proved obstruction" : "No verified circuit";
    status.className = "status obstruction";
    status.textContent = label + (mode.reason ? " · " + mode.reason : "");
    if (exact.checked && projectiveMode && projectiveMode.status === "verified_symbolic") {
      status.append(" ", element("span", "Clear “Require exact phase” to see the projective circuit."));
    }
    drawBases();
    drawQasm();
    return;
  }
  status.className = "status";
  status.textContent = reusesExact ? "Verified · Exact circuit reused" :
    exact.checked ? "Verified · Exact phase" : "Verified · Up to global phase";
  counts.append(
    element("span", numberLabel(mode.entanglers, "entangler", "entanglers")),
    element("span", numberLabel(mode.locals, "single-qubit gate", "single-qubit gates")),
  );
  const phase = byId("phase-note");
  phase.hidden = mode.factor === undefined;
  if (!phase.hidden) {
    phase.replaceChildren("Virtual phase in the atlas record ", element("code", data.expressions[mode.factor][0]),
      ". Multiply the circuit matrix by this factor to recover the target.");
  }
  const operations = mode.operations;
  byId("operation-count").textContent = numberLabel(target.width, "qubit", "qubits") +
    " · " + numberLabel(operations.length, "operation", "operations");
  drawCircuit(target.width, operations, native.entangler);
  const details = byId("operation-details");
  details.open = false;
  details.hidden = !operations.length;
  drawOperations(operations, mode.bindings);
  const canExportSymbolic = supportsSymbolic(mode, target, data.expressions);
  const previousValues = resultInputs(gate);
  fields.replaceChildren(...inputs.map((name, index) => {
    const label = element("label", name);
    const input = element("input");
    input.type = "number";
    input.step = "any";
    input.name = name;
    input.value = previousValues.get(name) ?? defaultAngle(index);
    label.append(input);
    return label;
  }));
  fields.dataset.gate = gate;
  format.options[0].disabled = !canExportSymbolic;
  byId("format-label").hidden = !inputs.length || !canExportSymbolic;
  format.value = !canExportSymbolic || (inputs.length && previousValues.size && format.value === "bound")
    ? "bound" : "symbolic";
  result = {gate, basis, mode, target, canExportSymbolic};
  drawBases();
  drawQasm();
}

function resultInputs(gate) {
  return new Map(fields.dataset.gate === gate
    ? Array.from(fields.querySelectorAll("input"), input => [input.name, input.value])
    : []);
}

async function copyQasm() {
  if (!qasm) return;
  const source = qasm;
  try {
    await navigator.clipboard.writeText(source);
  } catch {
    const temp = element("textarea", source, "sr-only");
    document.body.append(temp);
    temp.select();
    const copied = document.execCommand("copy");
    temp.remove();
    copy.focus();
    if (!copied) {
      byId("qasm").focus();
      const range = document.createRange();
      range.selectNodeContents(byId("qasm"));
      const selection = getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      byId("copy-status").textContent = "QASM selected. Use your system copy shortcut.";
      return;
    }
  }
  if (source === qasm) copy.textContent = "Copied";
  byId("copy-status").textContent = "Copied OpenQASM.";
}

function downloadQasm() {
  if (!qasm) return;
  const file = URL.createObjectURL(new Blob([qasm], {type: "text/plain"}));
  const link = element("a");
  link.href = file;
  link.download = result.gate + "-" + result.basis +
    (exact.checked ? "-exact" : "-projective") + ".qasm";
  link.click();
  setTimeout(() => URL.revokeObjectURL(file), 1000);
}

async function start() {
  const encoded = byId("atlas-data").textContent.trim();
  const compressed = Uint8Array.from(atob(encoded), char => char.charCodeAt(0));
  data = await new Response(new Blob([compressed]).stream()
    .pipeThrough(new DecompressionStream("gzip"))).json();
  options(gateSelect, Object.keys(data.targets).sort((a, b) => a.localeCompare(b)));
  options(basisSelect, data.bases.map(basis => basis.name));
  const initial = new URLSearchParams(location.search);
  const requestedGate = initial.get("gate");
  const requestedBasis = initial.get("basis");
  gateSelect.value = Object.hasOwn(data.targets, requestedGate) ? requestedGate : "CCX";
  basisSelect.value = data.bases.some(basis => basis.name === requestedBasis) ? requestedBasis : "cx_u";
  exact.checked = initial.get("phase") !== "projective";
  byId("provenance").textContent = "PDF atlas archive · Source revision " + data.source +
    " · Data SHA-256 " + data.sha256;
  for (const control of [gateSelect, basisSelect, exact]) control.addEventListener("change", showSelection);
  fields.addEventListener("input", drawQasm);
  format.addEventListener("change", drawQasm);
  basisSearch.addEventListener("input", drawBases);
  copy.addEventListener("click", copyQasm);
  download.addEventListener("click", downloadQasm);
  showSelection();
  byId("loading").hidden = true;
  byId("explorer").hidden = false;
}

start().catch(error => {
  byId("loading").textContent = "Could not load the atlas. " + error.message;
});
