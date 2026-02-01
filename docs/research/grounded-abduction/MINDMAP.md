# Mind Map: Grounded Abduction

## Core
```
Grounded Abduction
├── The Abduction Gap
│   ├── "LLMs Can't Jump" (Zahavy 2026) → formalizes via Peirce's Taxonomy
│   ├── Frankfurt's Problem → invention ≠ selection of hypotheses
│   ├── Creativity as Compression → fails with no error signal
│   └── Chinese Room → symbols without physical grounding
│
├── Inference Taxonomy (Peirce 1934)
│   ├── Deduction: Rule+Case→Result → AlphaProof, Lean
│   ├── Induction: Case+Result→Rule → LLMs, FunSearch
│   └── Abduction: Rule+Result→Case → THIS PROJECT
│       └── Manipulative Abduction (Magnani) → requires embodied sim
│
├── Existing Systems (Each Insufficient)
│   ├── Frozen LLM Search: AlphaEvolve, FunSearch, ShinkaEvolve, ThetaEvolve
│   ├── Context Engineering: ACE, Dynamic Cheatsheet
│   ├── Test-Time Learning: TTT-Discover, TTRL, EvoTune
│   └── Physics Discovery: AI Physicist, SGA, AI-Newton, PySR
│
├── Our Three-Level Optimization
│   ├── Level 1 (Inner): Newton → gradient-based param optimization
│   ├── Level 2 (Middle): ACE → playbook-based strategy optimization
│   └── Level 3 (Outer): ShinkaEvolve → population evolution
│
└── Math Foundations
    ├── Eluder Dimension → O(√(d_elu T)) for ACE exploration
    ├── ACR Framework → adaptive mutation = linear convergence
    ├── PAC-Bayes → KL(Q||P) for program generalization
    ├── Bilevel Optimization → O(t^{-1/2}) convergence
    ├── OCO with Memory → O(√(H_p T)) for playbook history
    ├── Pearl's 3-Step = Formal Abduction
    └── Gumbel-Softmax + REINFORCE → gradient bridge
```

## Connections
- [[Manipulative Abduction]] ←requires→ [[Newton]]: embodied simulation substrate
- [[ACE Playbook]] ←replaces→ [[ShinkaEvolve task_sys_msg]]: evolving strategies
- [[Pearl's do(X)]] ←formalizes→ [[Manipulative Abduction]]: computational "thinking by doing"
- [[Active Causal Discovery]] ←operationalizes→ [[Abductive Loop]]: POMDP + info-gain
- [[SGA Bilevel]] ←subset of→ [[Our Three-Level]]: SGA=L1+L3, we add ACE=L2
- [[EvoTune DPO]] ←solves→ [[Frozen LLM Problem]]: preference learning
- [[Eluder Dimension]] ←bounds→ [[ACE Exploration]]: independent of bullet count
- [[Creativity as Compression]] ←contradicted by→ [[Einstein's GR]]: no error signal
- [[Gumbel-Softmax]] ←bridges→ [[Discrete↔Continuous]]: program token relaxation

## Gaps
- [ ] Formalize three-level convergence (our contribution)
- [ ] Prove ACE+ShinkaEvolve regret bound vs ShinkaEvolve alone
- [ ] Build Newton physics discovery benchmark
- [ ] Validate Q-estimator alternatives on physics tasks
- [ ] Adapt TTT-Discover to physical law discovery

## Learning Path
1. **Start**: Peirce abduction + Magnani manipulative abduction + "LLMs Can't Jump"
2. **Then**: AlphaEvolve/ShinkaEvolve (evolutionary search) + ACE (context engineering)
3. **Then**: Pearl's do-calculus + SGA bilevel + gradient estimators
4. **Then**: Eluder dimension, PAC-Bayes, ACR, OCO with memory
5. **Deep dive**: Formalize three-level optimization with convergence
6. **Build**: Newton environments + ShinkaEvolve×ACE integration
7. **Evaluate**: Frozen vs learning, ACE vs no-ACE, Q-estimator variants
