const qasmGates = {
  U3: "U", H: "h", X: "x", S: "s", Sx: "sx", T: "t", Tdg: "tdg",
  P: "p", Rx: "rx", Ry: "ry", Rz: "rz", CX: "cx", CZ: "cz",
};

const nativeDefinitions = {
  Rzz: "atlas_rzz(t) a, b { cx a, b; rz(t) b; cx a, b; }",
  Rxx: "atlas_rxx(t) a, b { h a; h b; atlas_rzz(t) a, b; h a; h b; }",
  Ryy: "atlas_ryy(t) a, b { sdg a; sdg b; atlas_rxx(t) a, b; s a; s b; }",
  Rzx: "atlas_rzx(t) a, b { h b; atlas_rzz(t) a, b; h b; }",
  R: "atlas_r(t, p0) a { rz(-p0) a; rx(t) a; rz(p0) a; }",
  PhasedX: "atlas_phasedx(t, p0) a { atlas_r(pi*t, pi*p0) a; }",
  PhasedXZ: "atlas_phasedxz(x0, z0, a0) a { p(-pi*a0) a; rx(pi*x0) a; p(pi*(z0+a0)) a; gphase(pi*x0/2); }",
  GPi: "atlas_gpi(p0) a { rz(-2*pi*p0) a; x a; rz(2*pi*p0) a; }",
  GPi2: "atlas_gpi2(p0) a { atlas_r(pi/2, 2*pi*p0) a; }",
  ECR: "atlas_ecr a, b { gphase(-pi/4); s a; sx b; cx a, b; x a; }",
  fSim: "atlas_fsim(t, p0) a, b { atlas_rxx(t) a, b; atlas_ryy(t) a, b; cp(-p0) a, b; }",
  SYC: "atlas_syc a, b { atlas_fsim(pi/2, pi/6) a, b; }",
  iSWAP: "atlas_iswap a, b { atlas_rxx(-pi/2) a, b; atlas_ryy(-pi/2) a, b; }",
  SiSWAP: "atlas_siswap a, b { atlas_rxx(-pi/4) a, b; atlas_ryy(-pi/4) a, b; }",
  SSWAP: "atlas_sswap a, b { gphase(pi/8); atlas_rxx(pi/4) a, b; atlas_ryy(pi/4) a, b; atlas_rzz(pi/4) a, b; }",
  B: "atlas_b a, b { atlas_rxx(-pi/2) a, b; atlas_ryy(-pi/4) a, b; }",
  XX90: "atlas_xx90 a, b { atlas_rxx(pi/2) a, b; }",
  ZZMax: "atlas_zzmax a, b { atlas_rzz(pi/2) a, b; }",
  ZZPhase: "atlas_zzphase(t) a, b { atlas_rzz(pi*t) a, b; }",
  TK2: "atlas_tk2(t1, t2, t3) a, b { atlas_rxx(pi*t1) a, b; atlas_ryy(pi*t2) a, b; atlas_rzz(pi*t3) a, b; }",
  IonQMS: "atlas_ionqms(p0, p1, t) a, b { rz(-2*pi*p0) a; rz(-2*pi*p1) b; atlas_rxx(2*pi*t) a, b; rz(2*pi*p0) a; rz(2*pi*p1) b; }",
  IonQMSMax: "atlas_ionqmsmax(p0, p1) a, b { atlas_ionqms(p0, p1, 1.0/4) a, b; }",
  IonQZZ: "atlas_ionqzz(t) a, b { atlas_rzz(2*pi*t) a, b; }",
};

const symbolicFunctions = {sin: "sin", cos: "cos", sqrt: "sqrt", atan: "arctan", acos: "arccos", exp: "exp", log: "log"};

function qasmNumber(value) {
  if (!Number.isFinite(value)) throw Error("An angle is undefined at these parameter values.");
  const [mantissa, exponent] = String(value).split("e");
  const decimal = mantissa.includes(".") ? mantissa : mantissa + ".0";
  return exponent === undefined ? decimal : decimal + "e" + exponent;
}

function symbolicAngle(node, inputs) {
  if (typeof node === "number") return qasmNumber(node);
  if (node === "pi") return node;
  if (typeof node === "string" && inputs.includes(node)) return "p_" + node;
  if (!Array.isArray(node)) throw Error("Unsupported symbolic expression");
  const [op, left, right] = node;
  if (op === "-" && node.length === 2) return "(-" + symbolicAngle(left, inputs) + ")";
  if (["+", "-", "*", "/", "**"].includes(op)) {
    return "(" + symbolicAngle(left, inputs) + " " + op + " " + symbolicAngle(right, inputs) + ")";
  }
  if (Object.hasOwn(symbolicFunctions, op)) return symbolicFunctions[op] + "(" + symbolicAngle(left, inputs) + ")";
  throw Error("Unsupported symbolic expression");
}

function supportsSymbolic(mode, target, expressions) {
  if (mode.bindings) return false;
  try {
    for (const [, , parameters] of mode.operations) {
      for (const index of parameters) symbolicAngle(expressions[index][1], target.inputs);
    }
    return true;
  } catch {
    return false;
  }
}

function real(value) {
  if (typeof value === "number") return value;
  if (Math.abs(value[1]) > 1e-9 * Math.max(1, Math.abs(value[0]))) {
    throw Error("An angle evaluated to a complex value.");
  }
  return value[0];
}

function parts(value) {
  return typeof value === "number" ? [value, 0] : value;
}

function complexExp([re, im]) {
  const radius = Math.exp(re);
  return [radius * Math.cos(im), radius * Math.sin(im)];
}

function complexLog([re, im]) {
  return [Math.log(Math.hypot(re, im)), Math.atan2(im, re)];
}

function arithmetic(op, left, right) {
  if (typeof left === "number" && typeof right === "number") {
    if (op === "+") return left + right;
    if (op === "-") return left - right;
    if (op === "*") return left * right;
    if (op === "/") return left / right;
    if (left >= 0 || Number.isInteger(right)) return left ** right;
  }
  const [a, b] = parts(left);
  const [c, d] = parts(right);
  switch (op) {
    case "+": return [a + c, b + d];
    case "-": return [a - c, b - d];
    case "*": return [a * c - b * d, a * d + b * c];
    case "/": {
      const square = c * c + d * d;
      return [(a * c + b * d) / square, (b * c - a * d) / square];
    }
    case "**": return complexExp(arithmetic("*", complexLog([a, b]), right));
    default: throw Error("Unknown arithmetic operation");
  }
}

function equal(left, right) {
  const [a, b] = parts(left);
  const [c, d] = parts(right);
  return a === c && b === d;
}

function angleEvaluator(expressions, bindings, inputs) {
  const names = new Map([["pi", Math.PI], ["I", [0, 1]], ...inputs]);
  const definitions = new Map(bindings || []);
  function evaluate(node) {
    if (typeof node === "number" || typeof node === "boolean") return node;
    if (typeof node === "string") {
      if (names.has(node)) return names.get(node);
      if (!definitions.has(node)) throw Error("No value for " + node + ".");
      const value = evaluate(expressions[definitions.get(node)][1]);
      names.set(node, value);
      return value;
    }
    const [op, first, second] = node;
    if (op === "Piecewise") {
      for (const [value, condition] of node.slice(1)) {
        if (evaluate(condition)) return evaluate(value);
      }
      throw Error("No formula branch applies at these parameter values.");
    }
    const left = evaluate(first);
    if (op === "&") return left && evaluate(second);
    if (op === "~") return !left;
    if (node.length === 2) {
      switch (op) {
        case "-": return arithmetic("*", -1, left);
        case "sqrt": return arithmetic("**", left, 0.5);
        case "exp": return typeof left === "number" ? Math.exp(left) : complexExp(left);
        case "log": return typeof left === "number" && left >= 0 ? Math.log(left) : complexLog(parts(left));
        case "Abs": return Math.hypot(...parts(left));
        case "re": return parts(left)[0];
        case "im": return parts(left)[1];
        case "conjugate": return [parts(left)[0], -parts(left)[1]];
        case "sin": return Math.sin(real(left));
        case "cos": return Math.cos(real(left));
        case "atan": return Math.atan(real(left));
        case "acos": return Math.acos(real(left));
        case "floor": return Math.floor(real(left));
        case "sinc": {
          const angle = real(left);
          return angle === 0 ? 1 : Math.sin(angle) / angle;
        }
        default: throw Error("Unknown expression " + op + ".");
      }
    }
    const right = evaluate(second);
    switch (op) {
      case "+": case "-": case "*": case "/": case "**": return arithmetic(op, left, right);
      case "<": return real(left) < real(right);
      case "<=": return real(left) <= real(right);
      case ">": return real(left) > real(right);
      case ">=": return real(left) >= real(right);
      case "Eq": return equal(left, right);
      case "Ne": return !equal(left, right);
      case "atan2": return Math.atan2(real(left), real(right));
      case "Mod": return real(left) - real(right) * Math.floor(real(left) / real(right));
      default: throw Error("Unknown expression " + op + ".");
    }
  }
  return index => evaluate(expressions[index][1]);
}

function exportQasm(name, basis, mode, target, expressions, format, values) {
  const symbolic = format === "symbolic";
  const parameterized = symbolic && target.inputs.length > 0;
  const usedDefinitions = new Set();
  const output = ["OPENQASM 3.0;", 'include "stdgates.inc";', "", "// " + name + " in " + basis];
  const evaluate = angleEvaluator(expressions, mode.bindings, values);
  if (mode.factor !== undefined) {
    output.push("// Up to global phase. Multiply the circuit matrix by the atlas factor.");
    if (symbolic) {
      output.push("// Atlas factor: " + expressions[mode.factor][0]);
    } else {
      const phase = Math.atan2(...parts(evaluate(mode.factor)).reverse());
      output.push("// Atlas factor: exp(i * " + qasmNumber(phase) + ")");
    }
  }
  function useGate(gate) {
    if (Object.hasOwn(qasmGates, gate)) return qasmGates[gate];
    const definition = nativeDefinitions[gate];
    if (!definition) throw Error("No OpenQASM definition for " + gate + ".");
    if (!usedDefinitions.has(definition)) {
      const dependencies = definition.split("{")[1].match(/\batlas_[a-z0-9]+\b/g) || [];
      for (const dependency of dependencies) {
        const key = Object.keys(nativeDefinitions).find(candidate =>
          nativeDefinitions[candidate].startsWith(dependency + "(") ||
          nativeDefinitions[candidate].startsWith(dependency + " "));
        useGate(key);
      }
      usedDefinitions.add(definition);
    }
    return definition.match(/^\w+/)[0];
  }
  const calls = mode.operations.map(([gate, wires, parameters]) => {
    const name = useGate(gate);
    const args = parameters.map(index => symbolic
      ? symbolicAngle(expressions[index][1], target.inputs)
      : qasmNumber(real(evaluate(index))));
    return name + (args.length ? "(" + args.join(", ") + ")" : "") + " " +
      wires.map(wire => parameterized ? "w" + wire : "q[" + wire + "]").join(", ") + ";";
  });
  if (usedDefinitions.size) output.push("", ...Array.from(usedDefinitions, line =>
    "gate " + line.replace(" { ", " {\n  ").replace(/; (?!})/g, ";\n  ").replace("; }", ";\n}")));
  const wires = Array.from({length: target.width}, (_, index) => index);
  if (parameterized) {
    const parameters = target.inputs.map(input => "p_" + input).join(", ");
    output.push("", "gate atlas_circuit(" + parameters + ") " + wires.map(index => "w" + index).join(", ") + " {",
      ...calls.map(call => "  " + call), "}", "", "qubit[" + target.width + "] q;",
      "// Example call (replace values as needed)",
      "// atlas_circuit(" + target.inputs.map((_, index) => qasmNumber(defaultAngle(index))).join(", ") +
      ") " + wires.map(index => "q[" + index + "]").join(", ") + ";");
  } else {
    output.push("", "qubit[" + target.width + "] q;", ...calls);
  }
  return output.join("\n") + "\n";
}

function defaultAngle(index) {
  return [0.37, 0.23, 0.61, 0.17][index % 4];
}
