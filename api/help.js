/* Shared explanations for the lab and chat. No model settings change here. */
(() => {
  const explanations = {
    groundedMode: ["Evidence mode", "Retrieves relevant passages from a frozen public portfolio and BB8 repository corpus before generation.", "Turning it on asks the model to cite evidence and abstain when information is missing. It does not guarantee factual accuracy. Expand the supplied evidence under each reply to check the sources. Use a grounded model for the evaluated 768-token budget."],
    chatModel: ["Chat model", "Chooses between the configured BB8 model and operator-enabled comparison models.", "Qwen Instruct is an external pretrained baseline, not a BB8-trained checkpoint. Changing the selection clears conversation history. Each model uses its matching prompt format; both can make errors."],
    apiKey: ["Local API key", "Authenticates requests to your local server.", "Changing it reconnects only when it matches the key used to start the server. It does not change predictions."],
    modelSelect: ["Trained model", "Chooses a saved checkpoint and its matching tokenizer.", "Changing models can change knowledge, vocabulary and context limits. It clears the current unsaved continuation; save it first to compare later."],
    strategy: ["Token selection", "Determines how the next token is selected from model scores.", "Greedy takes the highest score. Temperature samples across the vocabulary. Top-K limits the candidate count; Top-P limits cumulative probability. Sampling can produce different answers, including incorrect ones."],
    temperature: ["Temperature", "Rescales scores before sampling. A value of 1 leaves their scale unchanged.", "Lower values concentrate probability on higher scores; higher values spread it across more tokens. It has no effect with Greedy. A lower value does not guarantee a correct answer."],
    topP: ["Top-P threshold", "Keeps the smallest group of highest-probability tokens whose total reaches P, then samples within that group.", "Lower P usually narrows the choices; higher P allows more alternatives. P = 1 keeps the full distribution. Active only with Top-P; temperature also affects the group."],
    topK: ["Top-K candidates", "Keeps the K highest-scoring tokens before sampling.", "Smaller K narrows the choices; larger K allows more alternatives. K = 1 selects the highest score. Active only with Top-K; K is capped by the vocabulary size."],
    penalty: ["Repetition penalty", "Adjusts scores for tokens already in the visible context, including your prompt.", "1 applies no penalty. Raising it discourages those tokens relative to unseen tokens, but can also suppress useful repeated words or numbers. It applies in every strategy, including Greedy."],
    seed: ["Random seed", "Sets the starting point of the sampler’s random draws.", "Change it to explore another sampled continuation. Keep it fixed when comparing settings. The same input, model, settings and runtime should reproduce draws; Greedy is unaffected by the seed."],
    contextWindow: ["Context tokens", "Limits how many recent tokens the lab sends to the model at each step.", "Lower values discard earlier text sooner and may remove instructions. Higher values preserve more context up to this checkpoint’s limit. This affects the visible input, not the number of new tokens generated."],
    batchSteps: ["Tokens per run", "Sets how many new tokens Run selected count attempts in one click.", "A larger count generates a longer continuation and takes more time. Generation still stops at the end-of-text token or the lab’s 64-token limit. This is not a training batch size."],
    prompt: ["Your input", "Text the model receives before generating its response.", "Specific questions give clearer instructions than a topic such as France. In the lab, editing input clears the unsaved continuation; click Inspect prompt to calculate new probabilities."],
    promptMode: ["Prompt format", "Raw mode continues your text directly. Instruction mode wraps it in the selected model’s question-and-response format.", "Use instruction mode for questions, or raw mode to study text completion. The wrapper changes the input tokens and their predictions; expand Exact formatted input to see it."],
    runLabel: ["Experiment label", "A name for a saved inference experiment.", "Use a descriptive name such as France — temperature 0.5. Labels help compare runs and do not affect generation."],
    contextMode: ["Question context", "Fresh question sends your latest message. Conversation also sends earlier turns that fit the model’s context.", "Fresh question helps isolate a test. Conversation allows follow-up questions but can repeat earlier errors. Starting a new conversation clears the displayed history."],
    maxTokens: ["Maximum new tokens", "Caps the length of the generated response. A token can be a word, part of a word, punctuation or a special marker.", "Raising the cap allows longer responses and takes more time. Lowering it may cut an answer off mid-sentence. The model may finish before reaching the cap."],
    eligible: ["Eligible tokens", "Vocabulary IDs with nonzero probability in the final selection distribution.", "Top-K, Top-P and temperature can change this count. Greedy leaves one candidate. A manually forced token may be outside the eligible set."],
    rawEntropy: ["Raw entropy", "Measures how spread out the original model probabilities are, in bits.", "Lower entropy means the model favors fewer tokens. Higher entropy means more uncertainty about the next token. Decoding controls do not change this raw value when the model and context stay fixed. It is not a measure of factual accuracy."],
    selectionEntropy: ["Selection entropy", "Measures how spread out probabilities are after penalties, temperature and filtering.", "Lower temperature or narrower filters generally concentrate selection. Greedy has zero selection entropy. Compare this with raw entropy to see the effect of your controls."],
    contextUsed: ["Model context", "Shows the number of tokens visible to the model and the selected window limit.", "Once the window fills, the lab drops the earliest token IDs as new ones arrive. Expand the visible-context section to see exactly what remains."],
    rawLogit: ["Raw logit", "A score produced by the model for a vocabulary token before decoding controls. Scores can be negative and are not percentages.", "Higher scores indicate more likely next tokens relative to other scores at the same step. Changing the prompt or model changes these scores; sampling controls leave them unchanged."],
    modelProbability: ["Model probability", "Softmax converts the raw scores into probabilities that sum to 100% across the full vocabulary.", "This is the starting distribution. The Other tokens row includes probability mass outside the displayed candidates. A high probability does not prove a statement is true."],
    adjustedProbability: ["Adjusted probability", "Probability after repetition penalty and temperature, before Top-K or Top-P filtering.", "Compare this column with Model to see how the controls reshape predictions. With Greedy, temperature is skipped but repetition penalty still applies."],
    selectionProbability: ["Selection probability", "The distribution used to choose the next token after filtering and renormalizing.", "Excluded tokens have zero probability. Greedy assigns 100% to one token. Choose manually overrides this distribution, so you can explore a token the sampler would not select."]
  };

  let active = null, pinned = false, timer;
  function hide() {
    clearTimeout(timer);
    if (active) { active.tip.hidden = true; active.button.setAttribute("aria-expanded", "false"); }
    active = null; pinned = false;
  }
  function position() {
    if (!active) return;
    const box = active.button.getBoundingClientRect(), tip = active.tip;
    const width = tip.offsetWidth, height = tip.offsetHeight;
    tip.style.left = Math.max(12, Math.min(box.left, window.innerWidth - width - 12)) + "px";
    const below = box.bottom + 8;
    tip.style.top = Math.max(12, below + height <= window.innerHeight - 12 ? below : box.top - height - 8) + "px";
  }
  function show(item) {
    clearTimeout(timer);
    if (active !== item) { hide(); active = item; }
    item.tip.hidden = false; item.button.setAttribute("aria-expanded", "true"); position();
  }
  function scheduleHide(item) {
    clearTimeout(timer);
    timer = setTimeout(() => {
      if (active === item && !pinned && document.activeElement !== item.button &&
          !item.tip.matches(":hover") && !item.button.matches(":hover")) hide();
    }, 160);
  }

  for (const [id, [title, meaning, effect]] of Object.entries(explanations)) {
    const control = document.getElementById(id);
    const label = document.querySelector(`label[for="${id}"]`);
    const metric = document.querySelector(`[data-help-for="${id}"]`);
    if (!control || (!label && !metric)) continue;
    const button = document.createElement("button");
    button.type = "button"; button.className = "help-button"; button.textContent = "i";
    button.setAttribute("aria-label", "About " + title);
    button.setAttribute("aria-expanded", "false");
    const tip = document.createElement("div");
    tip.id = `help-${id}`; tip.className = "help-tooltip"; tip.setAttribute("role", "tooltip"); tip.hidden = true;
    const heading = document.createElement("strong"), description = document.createElement("p"), change = document.createElement("p");
    heading.textContent = title; description.textContent = meaning; change.textContent = effect;
    tip.append(heading, description, change);
    const descriptionId = tip.id;
    button.setAttribute("aria-describedby", descriptionId);
    control.setAttribute("aria-describedby", [control.getAttribute("aria-describedby"), descriptionId].filter(Boolean).join(" "));
    if (label) {
      const row = document.createElement("div"); row.className = "field-label";
      label.before(row); row.append(label, button);
    } else metric.append(button);
    document.body.append(tip);
    const item = {button, tip};
    button.addEventListener("pointerenter", () => { if (!pinned) show(item); });
    button.addEventListener("focus", () => show(item));
    button.addEventListener("click", () => {
      if (active === item && pinned) hide();
      else { show(item); pinned = true; }
    });
    button.addEventListener("pointerleave", () => scheduleHide(item));
    button.addEventListener("blur", () => { if (active === item) pinned = false; scheduleHide(item); });
    tip.addEventListener("pointerenter", () => clearTimeout(timer));
    tip.addEventListener("pointerleave", () => scheduleHide(item));
  }
  document.addEventListener("keydown", event => { if (event.key === "Escape") hide(); });
  document.addEventListener("pointerdown", event => {
    if (active && !active.button.contains(event.target) && !active.tip.contains(event.target)) hide();
  });
  window.addEventListener("resize", position);
  document.addEventListener("scroll", position, true);
})();
