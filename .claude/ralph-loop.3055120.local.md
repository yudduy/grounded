---
active: true
iteration: 1
max_iterations: 100
completion_promise: "RESEARCH_EXHAUSTED"
started_at: "2026-01-31T20:14:49Z"
---

Exhaustively research {topic}. You are a research orchestrator.

## RESEARCH PROTOCOL

Before adding any item to KNOWLEDGE.md, verify it:
1. STATE the finding explicitly
2. CITE the source (URL, paper, or multiple corroborating sources)
3. ASSESS confidence: is this from a primary source, secondary summary, or inference?
4. CROSS-REFERENCE: does this connect to or contradict anything already in KNOWLEDGE.md?

When you find contradictory information from different sources:
- Document BOTH viewpoints with citations
- Note the contradiction explicitly in KNOWLEDGE.md
- Assess which source is more authoritative and why

## Each Iteration:
1. READ docs/research/{topic}/KNOWLEDGE.md for current state
2. IDENTIFY gaps using systematic decomposition:
   - What sub-topics have zero coverage?
   - What claims have only one source? (need 2+)
   - What connections between concepts are unexplored?
   - What recent developments (last 12 months) are missing?
3. PRIORITIZE gaps by impact:
   - Core concepts with missing definitions: HIGH
   - Seminal papers not yet read: HIGH
   - Implementation details: MEDIUM
   - Tangential connections: LOW
4. DELEGATE parallel Explore agents to fill HIGH-priority gaps:
   - WebSearch for papers, implementations, discussions
   - WebFetch to read actual content (not just summaries)
   - One agent per knowledge gap
5. SYNTHESIZE findings into KNOWLEDGE.md (compact bullet format)
6. UPDATE Connections section - every new concept should link to at least one existing concept
7. UPDATE progress checklist

## Output Rules:
- Use bullet lists, not tables (more compact)
- Cap to top 10 most relevant items per section (unless --exhaustive)
- No empty placeholders - only add items with real content
- One line per item where possible
- Every item must have a source citation

## Exhaustiveness Rules:
- For each paper found: read abstract AND methodology
- For each concept: find 2+ sources confirming definition
- For each implementation: verify it exists (check GitHub stars, recent commits)
- Follow citation chains: if Paper A cites Paper B, research Paper B
- Cross-reference: if Concept X relates to Y, ensure both documented

## Quality Gates:
- No paper listed without reading its abstract
- No concept without definition AND source
- No implementation without verified URL
- All gaps either filled or marked BLOCKED with reason
- Every concept connected to at least one other concept in Connections section
- Contradictions between sources explicitly noted

## Self-Assessment (each iteration):
Before deciding whether to continue or stop, answer:
- What is the single biggest gap remaining?
- Would filling that gap change the Summary section?
- Am I finding genuinely new information, or re-confirming what I already know?
If the last 2 iterations produced no substantive new findings, strongly consider stopping.

## Completion:
When ALL of these are true:
- Core concepts documented with multiple sources
- Seminal papers read and summarized
- Key implementations catalogued
- Research gaps explicitly identified
- Connections section maps relationships between concepts
- No obvious unexplored threads
THEN: <promise>RESEARCH_EXHAUSTED</promise>

If genuinely stuck: <promise>BLOCKED: [reason]</promise>
