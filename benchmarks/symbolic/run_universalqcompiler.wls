(* Native exact-expression runner. Invoke with wolframscript -file followed by
   jobs.json output.json /path/to/QI.m /path/to/UniversalQCompiler.m .
   Upstream project: https://github.com/Q-Compiler/UniversalQCompiler
   Uses its public API; this file is an independent interchange adapter. *)

args = Rest[$ScriptCommandLine];
If[Length[args] != 4, Print["Expected jobs.json output.json QI.m UniversalQCompiler.m"]; Exit[2]];
{inputFile, outputFile, qiFile, compilerFile} = args;
Get[qiFile]; Get[compilerFile];
bundle = Import[inputFile, "RawJSON"];

Clear[encode];
encode[x_Integer] := <|"op" -> "Integer", "value" -> ToString[x, InputForm]|>;
encode[x_Rational] := <|"op" -> "Rational", "args" -> {encode[Numerator[x]], encode[Denominator[x]]}|>;
encode[Pi] := <|"op" -> "Pi"|>;
encode[I] := <|"op" -> "I"|>;
encode[E] := <|"op" -> "E"|>;
encode[x_Real] := <|"op" -> "Float", "value" -> ToString[x, InputForm]|>;
encode[x_Complex] := <|"op" -> "Plus", "args" -> {encode[Re[x]],
    <|"op" -> "Times", "args" -> {<|"op" -> "I"|>, encode[Im[x]]}|>}|> /; Re[x] != 0;
encode[x_Complex] := <|"op" -> "Times", "args" -> {<|"op" -> "I"|>, encode[Im[x]]}|>;
encode[x_Symbol] := <|"op" -> "Symbol", "name" -> SymbolName[x]|>;
encode[x_] := <|"op" -> ToString[Head[x], InputForm], "args" -> (encode /@ (List @@ x))|>;

encodeGate[g_] := Switch[g[[1]],
  0, <|"tag" -> 0, "control" -> g[[2]], "target" -> g[[3]]|>,
  1|2|3, <|"tag" -> g[[1]], "angle" -> encode[g[[2]]], "wire" -> g[[3]]|>,
  _, <|"tag" -> g[[1]], "raw" -> ToString[g, InputForm]|>
];

runJob[job_] := Module[{target, variables, gates, elapsed, answer},
  variables = ToExpression /@ Keys[job["symbolic_variables"]];
  target = ToExpression[job["matrix_input"], InputForm];
  {elapsed, gates} = AbsoluteTiming[TimeConstrained[
    Block[{$Assumptions = Element[variables, Reals]},
      Catch[Switch[job["method"],
        "ZYZDec", ZYZDec[target],
        "DecUnitary2Qubits", DecUnitary2Qubits[target, Simp -> False, FullSimp -> True],
        "QSD", QSD[target, Null, Simp -> False, FullSimp -> True],
        _, $Failed]]], job["timeout_seconds"], $TimedOut]];
  answer = <|"id" -> job["id"], "seconds" -> elapsed, "method" -> job["method"]|>;
  If[ListQ[gates] && AllTrue[gates, ListQ[#] && Length[#] == 3 &],
    Join[answer, <|"status" -> "completed", "gates" -> (encodeGate /@ gates)|>],
    Join[answer, <|"status" -> If[gates === $TimedOut, "timeout", "failed"],
                  "diagnostic" -> ToString[gates, InputForm]|>]]
];

results = {};
Do[
  AppendTo[results, runJob[job]];
  Export[outputFile, <|"schema_version" -> 1, "source" -> bundle["source"],
      "wolfram_version" -> $Version, "results" -> results|>, "RawJSON"],
  {job, bundle["jobs"]}
];
