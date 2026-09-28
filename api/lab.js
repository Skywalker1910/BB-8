"use strict";
const $ = (id) => document.getElementById(id);
let models = [], state = null, busy = false, dataOffset = 0, dataTotal = 0;
let datasetLoaded = false, saved = [];
const element = (tag, className = "", text = "") => {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  return node;
};
const percent = (value) => (100 * value).toFixed(value < 0.001 ? 4 : 2) + "%";
const piece = (token) => token.piece.replaceAll("\n", "↵").replaceAll(" ", "␣") || "∅";

async function api(path, body = {}) {
  const response = await fetch("/lab/" + path, {
    method: "POST", headers: {"content-type": "application/json", "x-api-key": $("apiKey").value},
    body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request failed.");
  return result;
}

function controls() {
  for (const node of document.querySelectorAll("button:not(.help-button),input,select,textarea")) node.disabled = busy;
  if (busy) return;
  $("prepare").disabled = !models.length;
  $("temperature").disabled = $("strategy").value === "greedy";
  $("topK").disabled = $("strategy").value !== "top_k";
  $("topP").disabled = $("strategy").value !== "top_p";
  $("strategyHint").textContent = {
    greedy: "Greedy selects the highest score after repetition penalty. Temperature, Top-K and Top-P are inactive.",
    temperature: "Sample across the full vocabulary. Temperature and repetition penalty are active; K and P are inactive.",
    top_k: "Sample among the K highest scores. Temperature and repetition penalty also apply; Top-P is inactive.",
    top_p: "Sample from the smallest group reaching probability P. Temperature and repetition penalty also apply; Top-K is inactive."
  }[$("strategy").value];
  const ended = !state || state.traces.length >= 64 || state.traces.at(-1)?.eos;
  for (const id of ["refresh", "sample", "generateMany"]) $(id).disabled = !!ended;
  $("undo").disabled = !state?.traces.length;
  $("restart").disabled = !state;
  $("save").disabled = !state?.traces.length;
  $("prevPage").disabled = dataOffset === 0;
  $("nextPage").disabled = dataOffset + 12 >= dataTotal;
}

async function run(task) {
  if (busy) return;
  busy = true; controls(); $("error").hidden = true; $("status").textContent = "Working…";
  try { await task(); $("status").textContent = "Local model ready"; }
  catch (error) { $("error").textContent = error.message; $("error").hidden = false; $("status").textContent = "Check the message below"; }
  finally { busy = false; controls(); }
}
const action = (id, task) => $(id).addEventListener("click", () => run(task));

function panel(name) {
  document.querySelectorAll(".panel").forEach(node => node.hidden = node.id !== name);
  document.querySelectorAll(".tab").forEach(node => {
    node.classList.toggle("active", node.dataset.panel === name);
    node.setAttribute("aria-pressed", String(node.dataset.panel === name));
  });
}
document.querySelectorAll(".tab").forEach(node => node.addEventListener("click", () => {
  panel(node.dataset.panel);
  if (node.dataset.panel === "dataset" && !datasetLoaded) run(loadDataset);
  if (node.dataset.panel === "experiments") run(loadRuns);
}));

function currentSettings() {
  return {strategy: $("strategy").value, temperature: Number($("temperature").value),
    top_k: Number($("topK").value), top_p: Number($("topP").value),
    repetition_penalty: Number($("penalty").value), seed: Number($("seed").value)};
}
function clearPreview() {
  if (state) state.preview = null;
  $("previewLabel").textContent = "Recalculate to inspect";
  $("candidates").replaceChildren();
  for (const id of ["eligible", "rawEntropy", "selectionEntropy", "contextUsed"]) $(id).textContent = "—";
  $("candidateNote").textContent = "";
  $("visibleTokens").replaceChildren();
  $("contextNote").textContent = "";
}
function resetState() {
  state = null; clearPreview(); renderState(); controls();
  $("formattedInput").textContent = ""; $("inputTokens").replaceChildren();
  $("inputMeta").textContent = "Inspect the edited prompt to begin.";
}
function modelChanged() {
  const selected = models.find(model => model.name === $("modelSelect").value);
  const record = selected.record;
  const max = record.model?.max_seq_len || 256;
  $("contextWindow").max = max; $("contextWindow").value = max;
  const params = record.parameters?.total || record.model?.parameters;
  $("modelSummary").textContent = `${params?.toLocaleString()} parameters · ${record.base_model ? "pretrained + LoRA" : "trained from scratch"} · ${max} token window`;
  $("modelRecord").textContent = JSON.stringify(record, null, 2);
  resetState();
}
$("modelSelect").addEventListener("change", modelChanged);
$("prompt").addEventListener("input", resetState);
$("promptMode").addEventListener("change", resetState);
const examples = {
  fact: ["What is the capital of France?", "instruction"],
  math: ["What is 4 * 4?", "instruction"],
  ambiguous: ["France", "instruction"],
  continuation: ["The capital of France is", "raw"]
};
document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => {
  [$("prompt").value, $("promptMode").value] = examples[button.dataset.example];
  resetState(); $("prompt").focus();
}));
for (const id of ["strategy", "temperature", "topP", "topK", "penalty", "seed", "contextWindow"]) {
  $(id).addEventListener("input", () => { updateReadouts(); clearPreview(); controls(); });
}
function updateReadouts() {
  for (const id of ["temperature", "topP", "penalty"]) $(id + "Value").textContent = Number($(id).value).toFixed(2);
}

function tokenChip(token, className = "") {
  const chip = element("span", "token " + className, piece(token));
  chip.append(element("small", "", "#" + token.id));
  chip.title = JSON.stringify(token.text) + " · ID " + token.id;
  return chip;
}

async function preparePrompt() {
  const spec = {model: $("modelSelect").value, prompt: $("prompt").value, mode: $("promptMode").value};
  const prepared = await api("prepare", spec);
  state = {spec, prepared, ids: [...prepared.input_ids], traces: [], preview: null};
  $("formattedInput").textContent = prepared.formatted_prompt;
  $("inputMeta").textContent = prepared.input_ids.length + " input tokens · " + prepared.backend;
  renderState();
  await inspect();
}
async function inspect(forcedTokenId) {
  if (!state) throw new Error("Inspect a prompt first.");
  if (state.traces.length >= 64 || state.traces.at(-1)?.eos) { clearPreview(); $("previewLabel").textContent = "Generation stopped"; return; }
  const trace = await api("step", {
    model: state.spec.model, input_ids: state.ids, prompt_token_count: state.prepared.input_ids.length,
    step_index: state.traces.length, settings: currentSettings(),
    context_window: Number($("contextWindow").value), forced_token_id: forcedTokenId
  });
  state.preview = trace; renderPreview(); return trace;
}

function renderState() {
  const traces = state?.traces || [];
  $("output").textContent = traces.length ? traces.at(-1).continuation || "[special token]" : "Your generated text will appear here.";
  $("stepCount").textContent = `${traces.length} / 64 tokens`;
  $("generatedTokens").replaceChildren();
  traces.forEach((trace, index) => {
    const button = element("button", "token " + (trace.selection_method === "manual" ? "manual" : ""), piece(trace.chosen));
    button.title = `Step ${index + 1} · token ${trace.chosen.id} · click to branch here`;
    button.addEventListener("click", () => run(async () => {
      state.traces = state.traces.slice(0, index + 1);
      state.ids = [...state.prepared.input_ids, ...state.traces.map(t => t.chosen.id)];
      renderState(); await inspect();
    }));
    $("generatedTokens").append(button);
  });
}

function probabilityCell(value, name) {
  const cell = element("td", "", percent(value));
  const bar = element("div", "bar " + name), fill = element("span");
  fill.style.width = (100 * value) + "%"; bar.append(fill); cell.append(bar);
  return cell;
}
function renderPreview() {
  const trace = state.preview;
  $("previewLabel").textContent = `Next draw: ${piece(trace.chosen)} · ${trace.latency_ms} ms`;
  $("eligible").textContent = `${trace.eligible_count.toLocaleString()} / ${trace.vocab_size.toLocaleString()}`;
  $("rawEntropy").textContent = trace.entropy.raw.toFixed(3);
  $("selectionEntropy").textContent = trace.entropy.selection.toFixed(3);
  $("contextUsed").textContent = `${trace.context_ids.length} / ${trace.context_window}`;
  $("candidates").replaceChildren();
  trace.candidates.forEach(token => {
    const row = element("tr", token.id === trace.chosen.id ? "selected" : token.eligible ? "" : "filtered");
    const name = element("td", "token-name", piece(token)); name.append(element("div", "muted", "#" + token.id));
    row.append(name, element("td", "", token.logit.toFixed(3)));
    for (const key of ["raw", "adjusted", "selection"]) row.append(probabilityCell(token[key], key));
    const actionCell = element("td"), choose = element("button", "choose", "Choose");
    choose.title = "Manually force this token, even if it was filtered out";
    choose.addEventListener("click", () => run(async () => { await inspect(token.id); commit(); await inspect(); }));
    actionCell.append(choose); row.append(actionCell); $("candidates").append(row);
  });
  const other = element("tr"); other.append(element("td", "", "Other tokens"), element("td", "", "—"));
  for (const key of ["raw", "adjusted", "selection"]) other.append(probabilityCell(trace.other_mass[key], key));
  other.append(element("td")); $("candidates").append(other);
  $("candidateNote").textContent = `Showing ${trace.candidates.length} candidates: raw leaders, selection leaders and the drawn token. Bars use the full vocabulary mass, not just these rows. Choose overrides the sampler.`;
  const allTokens = [...state.prepared.tokens, ...state.traces.map(t => t.chosen)];
  $("inputTokens").replaceChildren(...state.prepared.tokens.map((t, i) => tokenChip(t, i < trace.dropped_tokens ? "dropped" : "")));
  $("visibleTokens").replaceChildren(...allTokens.slice(-trace.context_window).map(t => tokenChip(t)));
  $("contextNote").textContent = `${trace.dropped_tokens} earlier tokens are outside this rolling context window. The exact remaining token IDs above are used in the forward pass; moving this control does not retrain the model.`;
}
function commit() {
  const trace = state.preview;
  state.ids.push(trace.chosen.id); state.traces.push(trace); state.preview = null;
  renderState();
}
action("prepare", preparePrompt);
action("refresh", () => inspect());
action("sample", async () => { await inspect(); commit(); await inspect(); });
action("generateMany", async () => {
  const count = Number($("batchSteps").value);
  if (!Number.isInteger(count) || count < 1 || count > 32) throw new Error("Choose 1–32 tokens per run.");
  for (let i = 0; i < count && state.traces.length < 64 && !state.traces.at(-1)?.eos; i++) {
    await inspect(); commit(); $("status").textContent = `Generated ${i + 1} / ${count}`;
  }
  await inspect();
});
action("undo", async () => { state.traces.pop(); state.ids.pop(); renderState(); await inspect(); });
action("restart", preparePrompt);
action("save", async () => {
  const result = await api("save", {...state.spec, label: $("runLabel").value || "Untitled experiment",
    expected_ids: state.traces.map(t => t.chosen.id),
    schedule: state.traces.map(t => ({settings: t.settings, context_window: t.context_window,
      forced_token_id: t.selection_method === "manual" ? t.chosen.id : null}))});
  await loadRuns(); $("leftRun").value = result.id;
  panel("experiments"); $("replayResult").textContent = "Saved and verified against the generated token IDs.";
  await compareRuns();
});

async function loadDataset() {
  const data = await api("dataset", {query: $("datasetQuery").value, category: $("datasetCategory").value,
    split: $("datasetSplit").value, offset: dataOffset});
  dataTotal = data.total; datasetLoaded = true;
  if ($("datasetCategory").options.length === 1) data.categories.forEach(c => $("datasetCategory").add(new Option(c, c)));
  $("datasetMeta").textContent = `${data.total.toLocaleString()} matching records · source rows ${data.records.length ? dataOffset + 1 : 0}–${Math.min(dataOffset + 12, data.total)} · split: ${data.split_model}`;
  $("datasetManifest").textContent = JSON.stringify(data.manifest, null, 2);
  $("datasetRows").replaceChildren();
  data.records.forEach(record => {
    const button = element("button", "record", record.instruction);
    button.append(element("span", "", `Row ${record.index + 1} · ${record.category} · ${record.split}`));
    button.addEventListener("click", () => run(() => loadExample(record.index)));
    $("datasetRows").append(button);
  });
  if (!data.records.length) $("datasetRows").append(element("p", "muted", "No matching records."));
}
function detail(title, text, open = false) {
  const node = element("details"); node.open = open;
  node.append(element("summary", "", title), element("pre", "", text)); return node;
}
async function loadExample(index) {
  const data = await api("example", {index});
  const box = $("examplePanel"); box.replaceChildren();
  box.append(element("div", "eyebrow", `${data.split} · v004 preprocessing`), element("h2", "", data.raw.instruction));
  box.append(detail("Raw JSON record", JSON.stringify(data.raw, null, 2)));
  box.append(detail("Formatted training example · before truncation", data.formatted_example));
  box.append(detail("Reference response from Dolly", data.raw.response));
  box.append(element("p", "muted", `Prompt kept: ${data.prompt_tokens_kept}/${data.prompt_tokens_original} tokens. Response + EOS kept: ${data.response_tokens_kept}/${data.response_tokens_original}. Training limit: ${data.max_length}.`));
  if (!data.response_marker_preserved || !data.eos_preserved) box.append(element("p", "warning", "Historical preprocessing truncated this example. " + (!data.response_marker_preserved ? "The full response marker is not preserved. " : "") + (!data.eos_preserved ? "EOS was cut off." : "")));
  box.append(element("p", "muted", "Gold = supervised response tokens. Blue = prompt tokens masked with label −100. Labels align with tokens; the causal loss shifts targets by one position inside the model."));
  const tokens = element("div", "tokens");
  data.tokens.forEach(t => { const chip = tokenChip(t, t.supervised ? "supervised" : ""); chip.title += " · label " + t.label; tokens.append(chip); });
  box.append(tokens);
  const use = element("button", "primary", "Inspect this training prompt");
  use.addEventListener("click", () => run(async () => {
    $("modelSelect").value = data.preview_model; modelChanged();
    $("prompt").value = data.formatted_prompt; $("promptMode").value = "raw";
    panel("playground"); await preparePrompt();
  }));
  box.append(use, element("p", "muted", "Loads the full prompt and context without the target response. The lab's rolling inference window can differ from historical training truncation; inspect the visible tokens."));
}
action("searchData", async () => { dataOffset = 0; await loadDataset(); });
$("datasetQuery").addEventListener("keydown", event => { if (event.key === "Enter") run(async () => { dataOffset = 0; await loadDataset(); }); });
action("prevPage", async () => { dataOffset = Math.max(0, dataOffset - 12); await loadDataset(); });
action("nextPage", async () => { dataOffset += 12; await loadDataset(); });

async function loadRuns() {
  saved = (await api("runs")).runs;
  for (const id of ["leftRun", "rightRun"]) {
    const old = $(id).value; $(id).replaceChildren();
    saved.forEach(r => $(id).add(new Option(`${r.label} · ${r.model} · ${r.created_at.slice(0, 19)}`, r.id)));
    if (saved.some(r => r.id === old)) $(id).value = old;
  }
  if (saved.length > 1 && $("leftRun").value === $("rightRun").value) $("rightRun").selectedIndex = 1;
  if (!saved.length) $("replayResult").textContent = "Generate at least one token in Playground, then Verify & save.";
}
async function selectedRun(id) {
  if (!$(id).value) throw new Error("Save an experiment first.");
  return api("run", {id: $(id).value});
}
function runCard(record, label) {
  const card = element("article", "card");
  card.append(element("div", "eyebrow", label), element("h2", "", record.label), element("p", "muted", `${record.model} · ${record.generated_ids.length} tokens · ${record.engine}`));
  card.append(element("p", "comparison-output", record.output));
  card.append(detail("Original prompt", record.recipe.prompt, true));
  card.append(detail("Settings at each step", JSON.stringify(record.recipe.schedule, null, 2)));
  card.append(detail("Provenance & artifact hashes", JSON.stringify({id: record.id, created_at: record.created_at, artifact_hashes: record.artifact_hashes, source_hashes: record.source_hashes, runtime: record.runtime, git_commit: record.git_commit, git_dirty: record.git_dirty}, null, 2)));
  return card;
}
async function compareRuns() {
  const a = await selectedRun("leftRun"), b = await selectedRun("rightRun");
  $("comparison").replaceChildren(runCard(a, "EXPERIMENT A"), runCard(b, "EXPERIMENT B"));
  if (a.recipe.prompt !== b.recipe.prompt || a.recipe.mode !== b.recipe.mode) $("replayResult").textContent = "These runs have different inputs or prompt formats; this is not a controlled decoding comparison.";
  else if (a.model !== b.model) $("replayResult").textContent = "Same input across different models. Their tokenizers may differ; token probabilities and token counts are not directly comparable.";
}
action("refreshRuns", loadRuns);
action("compare", compareRuns);
action("replay", async () => {
  const result = await api("replay", {id: $("leftRun").value});
  await loadRuns(); $("rightRun").value = result.id;
  await compareRuns();
  $("replayResult").textContent = result.matches_original ? "Replay matched every generated token ID. A new replay record was saved." : "Replay differed. A new record was saved; compare the runtime and traces before assuming reproducibility.";
});
action("useSetup", async () => {
  const record = await selectedRun("leftRun");
  $("modelSelect").value = record.model; modelChanged();
  $("prompt").value = record.recipe.prompt; $("promptMode").value = record.recipe.mode;
  const initial = record.recipe.schedule[0], options = initial.settings;
  for (const [id, key] of Object.entries({strategy: "strategy", temperature: "temperature", topK: "top_k", topP: "top_p", penalty: "repetition_penalty", seed: "seed"})) $(id).value = options[key];
  $("contextWindow").value = initial.context_window; updateReadouts();
  panel("playground"); await preparePrompt();
});
action("exportRun", async () => {
  const record = await selectedRun("leftRun");
  const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 2)], {type: "application/json"}));
  const link = element("a"); link.href = url; link.download = record.id + ".json"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

async function connect() {
  const data = await api("catalog"); models = data.models;
  $("modelSelect").replaceChildren();
  models.forEach(model => $("modelSelect").add(new Option(model.name, model.name)));
  if (!models.length) throw new Error("No registered checkpoints are available.");
  $("modelSelect").value = models.some(m => m.name === data.default_model) ? data.default_model : models[0].name;
  modelChanged(); updateReadouts();
}
action("connect", connect);
run(connect);
