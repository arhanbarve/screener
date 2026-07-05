---
name: chat-handoff
description: Use when a chat conversation needs to continue in a fresh conversation with none of this history, e.g. hitting length limits, switching devices, wanting a clean context window, or resuming work from a handoff someone else wrote. Produces a downloadable HANDOFF-*.md file the person carries into the new chat, or reads one back in.
---

# Chat Handoff

## Overview

Transfer an in-progress chat task to a fresh conversation that has zero history: no memory of this thread, no artifacts already loaded, nothing except what the person pastes or uploads. There is no shared filesystem and no git between chats — the handoff file itself is the only channel.

**Core principle:** the receiver (a fresh Claude instance) has nothing but the handoff file and whatever the person re-uploads alongside it. Write down only what cannot be reconstructed by looking at those attachments, plus a concrete resume plan.

## Two Modes

Detect intent from the person's phrasing:

| Person says | Mode |
|---|---|
| "create a handoff", "make a handoff", "wrap this up so I can continue elsewhere", "I'm about to hit the limit", "summarize this so I can start fresh" | **WRITE** |
| "resume from handoff", "continue from this handoff", "pick up where I left off", pastes/uploads a HANDOFF file | **RESUME** |

If ambiguous, ask which one.

## WRITE Mode

Goal: produce a downloadable `HANDOFF-YYYY-MM-DD-HHMM.md` file the person can carry into a new chat.

### Steps

1. **Get a real timestamp** — run `date +%Y-%m-%d-%H%M` via bash_tool. Never guess or hardcode it.
2. **Inventory what actually exists in this conversation:**
   - Artifacts created (names/titles, and whether they're finished or mid-edit)
   - Files the person uploaded (names — the receiver won't have these unless re-uploaded)
   - Any tool state that doesn't survive the handoff (a connector that was authorized, a search already done, a doc already fetched)
3. **Fill the template** below. Every section is required; write `None` rather than deleting a section that doesn't apply.
4. **Create the file** with create_file, save to `/mnt/user-data/outputs/HANDOFF-<timestamp>.md`, and use present_files so the person can download it.
5. **Tell the person** what to do next: download the file, and in the new chat, upload the handoff file plus any source files/artifacts it references (list which ones by name).

### Handoff Template

```markdown
# Handoff — <timestamp>

## Mission
<2-3 lines: what we are trying to accomplish and why. The end goal, not the mechanics.>

## Status
- Done: <what is finished>
- In progress: <what is half-built, and exactly where it stands>
- Not started: <known remaining work>

## Attachments Needed on Resume
<Files, artifacts, or documents the receiver must have re-uploaded or re-attached to continue. Name each one and what it is. "None" if the conversation was self-contained.>

## Key Decisions & Rationale
<The "why" that exists only in this chat. Each decision: what we chose, what we rejected, and why. This is the highest-value section — it's the thing a fresh instance cannot infer from files alone.>

## Dead Ends
<Approaches already tried and abandoned, with the reason. Stops the receiver repeating them. "None" if genuinely none.>

## Preferences & Constraints Established This Chat
<Style, tone, formatting, scope, or other constraints the person set mid-conversation that aren't captured elsewhere (e.g. memory) and would otherwise be lost.>

## Next Steps
1. <concrete, ordered, actionable — the receiver should be able to start at step 1 without asking clarifying questions>
2. ...

## Open Questions
<Anything needing the person's decision before proceeding. "None" if none.>
```

## RESUME Mode

Goal: read an uploaded handoff file and continue the work as if this were the same conversation.

### Steps

1. **Read the handoff file in full** (view or file-reading skill as appropriate).
2. **Check the "Attachments Needed on Resume" section** against what's actually been uploaded in this conversation. If something's missing, ask for it before proceeding rather than guessing at its contents.
3. **Confirm understanding** to the person in a few lines: the mission, current status, and the next step about to be taken. Surface any open questions from the handoff before acting on them.
4. **Continue** from Next Steps.

## Common Mistakes

- **Hardcoding the date** instead of running `date`. Filenames must reflect real time so they sort correctly if the person collects several.
- **Restating content already in an attached file** (full document text, full artifact contents). Point to it by name instead.
- **Vague next steps** ("continue the analysis"). The receiver has no context beyond this file — steps must be executable cold.
- **Forgetting attachments.** The single most common failure mode of a chat handoff (vs. a git-based one) is the receiver not having the source files, because there's no shared storage to pull from. Always spell out what needs to be re-uploaded.
- **Dropping sections** that seem empty. Write `None` so the receiver knows it was considered, not forgotten.
- **Not actually generating the file.** Producing the handoff content as chat text isn't enough — it must be a real downloadable file, or it's lost when the conversation ends.
