# English AI-Writing Patterns

Reference for detecting and removing AI-generated patterns in English writing.

## Content Patterns

### 1. Undue Emphasis on Significance

**Watch for:** stands as, serves as, is a testament/reminder, pivotal moment, crucial role, underscores its importance, reflects broader, symbolizing its enduring, contributing to the, setting the stage for, marking/shaping the, represents a shift, evolving landscape, deeply rooted

**Before:**
> The new framework was released in 2024, marking a pivotal moment in the evolution of web development.

**After:**
> The new framework was released in 2024. It adds server-side rendering and cuts bundle sizes by 40%.

**Fix:** Remove significance claims. Replace with concrete facts, numbers, or specific features.

---

### 2. Vague Attributions and Weasel Words

**Watch for:** Industry reports, Observers have cited, Experts argue, Some critics argue, several sources (when few cited)

**Before:**
> Experts believe this approach will transform the industry.

**After:**
> In a 2024 survey of 500 engineers by Stack Overflow, 67% said this approach reduced debugging time.

**Fix:** Name the source, the year, and the specific finding. If no source exists, state it as your own opinion.

---

### 3. Superficial Analyses with -ing Endings

**Watch for:** highlighting, underscoring, emphasizing, ensuring, reflecting, symbolizing, contributing to, showcasing

**Before:**
> The app uses a dark mode, reflecting the team's commitment to accessibility and showcasing their user-centric design philosophy.

**After:**
> The app has a dark mode. The team added it after users complained about eye strain during night shifts.

**Fix:** Delete the "-ing" significance clause. Replace with the actual reason or context.

---

### 4. Promotional and Advertisement-like Language

**Watch for:** boasts a, vibrant, rich (figurative), profound, enhancing its, showcasing, exemplifies, commitment to, nestled, in the heart of, breathtaking, stunning, must-visit, groundbreaking (figurative)

**Before:**
> Nestled in the heart of the tech district, the office boasts a vibrant culture and breathtaking views.

**After:**
> The office is in the tech district, three blocks from the subway. The rooftop has a view of the bay.

**Fix:** Replace adjectives with facts. "Breathtaking views" → what can you actually see?

---

### 5. Outline-like "Challenges and Future Prospects" Sections

**Watch for:** Despite its..., faces several challenges..., Despite these challenges, Challenges and Legacy, Future Outlook

**Before:**
> Despite its popularity, the library faces several challenges typical of open-source projects, including maintainer burnout and funding gaps. Despite these challenges, it continues to thrive.

**After:**
> The library has 3 active maintainers, down from 12 in 2020. The core author left last year. A corporate sponsor stepped in this March to fund one full-time developer.

**Fix:** Replace formulaic challenge sections with specific, dated facts.

## Language and Grammar Patterns

### 6. Overused "AI Vocabulary" Words

**High-frequency AI words:** Additionally, align with, crucial, delve, emphasizing, enduring, enhance, fostering, garner, highlight (verb), interplay, intricate, key (adjective), landscape (abstract), pivotal, showcase, tapestry, testament, underscore, valuable, vibrant

**Also overused:** elevate, embark, harness, leverage, navigate, resonate, revolutionize, robust, unleash, unlock; plus stock modifiers cutting-edge, game-changer, next-level, seamless

**Before:**
> Additionally, the platform boasts an intricate tapestry of features, showcasing the team's commitment to fostering innovation.

**After:**
> The platform has real-time collaboration, version history, and API access. We built these because users kept asking for them.

**Fix:** Swap AI words for plain alternatives. "Additionally" → "Also" or just remove. "Showcasing" → delete and state the fact directly.

---

### 7. Copula Avoidance (Avoidance of "is"/"are")

**Watch for:** serves as, stands as, marks, represents [a], boasts, features, offers [a]

**Before:**
> The tool serves as a comprehensive solution for developers. It features over 50 integrations and boasts a 99.9% uptime guarantee.

**After:**
> The tool is a code formatter. It has 50 integrations and a 99.9% uptime guarantee.

**Fix:** Replace elaborate constructions with simple "is/has" statements.

---

### 8. Negative Parallelisms and Inversion Pivots

**Watch for:** Not only... but..., It's not just about..., it is..., Not merely... it is..., X wasn't Y. It was Z., X used to be Y, now it's Z, X has been a Y, but here it's Z

**Before:**
> It's not just about speed; it's about reliability. The bottleneck wasn't the algorithm. It was the network. Readability used to be a strength, but here it's a problem.

**After:**
> Speed matters, but reliability matters more. The bottleneck is the network. Readability makes this problem harder.

**Fix:** Cut the construction when the contrast is hollow, since AI leans on this rhythm to manufacture drama. But it's a legitimate rhetorical device (JFK: "ask not what your country can do for you..."), so keep it when the contrast is real and earned. The tell is empty, repeated reframing, not the form itself.

---

### 9. Rule of Three Overuse

**Before:**
> The course covers fundamentals, advanced techniques, and real-world applications. You will gain knowledge, confidence, and practical skills.

**After:**
> The course covers fundamentals and advanced techniques. You will build an actual project by the end.

**Fix:** If there are really three items, keep them. If you are forcing a third for symmetry, drop it.

---

### 10. Elegant Variation (Synonym Cycling)

**Before:**
> The developer faced many bugs. The programmer fixed the issues. The coder shipped the release. The engineer celebrated.

**After:**
> The developer faced many bugs, fixed them, shipped the release, and celebrated.

**Fix:** Use the same term consistently. Readers do not mind repetition as much as AI thinks.

---

### 11. False Ranges

**Watch for:** from X to Y constructions where X and Y are not on a meaningful scale

**Before:**
> Our journey has taken us from startup garages to Fortune 500 boardrooms, from Python scripts to distributed systems.

**After:**
> We started in a garage writing Python scripts. Now we serve three Fortune 500 clients with distributed systems.

**Fix:** Remove the "from... to" poetic framing. State the before and after as separate facts.

## Style Patterns

### 12. Em Dash Overuse

**Before:**
> The bug was not in the API — it was in the caching layer — yet the team spent two days looking elsewhere.

**After:**
> The bug was not in the API; it was in the caching layer. The team spent two days looking elsewhere.

**Fix:** The em dash is the strongest AI tell. Replace it with a comma, colon, parentheses, period, or semicolon.

---

### 13. Overuse of Boldface

**Before:**
> It blends **OKRs**, **KPIs**, and **visual strategy tools** such as the **Business Model Canvas**.

**After:**
> It blends OKRs, KPIs, and visual strategy tools like the Business Model Canvas.

**Fix:** Remove mechanical bolding. If emphasis is truly needed, use it sparingly.

---

### 14. Inline-Header Vertical Lists

**Before:**
> - **Speed:** Processing is 3x faster
> - **Security:** End-to-end encryption added
> - **Cost:** Monthly fees cut by 40%

**After:**
> Processing is 3x faster, end-to-end encryption is now standard, and monthly fees dropped 40%.

**Fix:** Convert bold-label lists into flowing prose or simple bullet points without labels.

---

### 15. Title Case in Headings

**Before:**
> ## Strategic Negotiations And Global Partnerships

**After:**
> ## Strategic negotiations and global partnerships

**Fix:** Use sentence case for headings, unless a project's house style requires otherwise.

---

### 16. Emojis in Text

**Before:**
> 🚀 Launch phase: Product ships in Q3. 💡 Key insight: Users prefer simplicity.

**After:**
> The product ships in Q3. User research showed a preference for simplicity.

**Fix:** Remove emojis from body text. They are fine in social or chat contexts, not in articles, docs, or formal prose.

---

### 17. Curly Quotation Marks

**Before:**
> He said "the project is on track" but others disagreed.

**After:**
> He said "the project is on track" but others disagreed.

**Fix:** Use straight quotes (") in technical and casual writing.

## Communication Patterns

### 18. Chat Artifacts

**Watch for:** I hope this helps, Of course!, Certainly!, You're absolutely right!, Would you like..., let me know, here is a...

**Before:**
> Here is an overview of the migration. I hope this helps! Let me know if you'd like me to expand.

**After:**
> This migration moves the database from PostgreSQL 13 to 15. It took three weeks and broke authentication twice.

**Fix:** Remove all conversational residue. Prose is not a chat log.

---

### 19. Knowledge-Cutoff Disclaimers

**Watch for:** as of [date], Up to my last training update, While specific details are limited..., based on available information...

**Before:**
> While specific details about the release are limited in available sources, it appears to be scheduled for Q2.

**After:**
> The release is scheduled for Q2, according to the team's public roadmap.

**Fix:** State the source or remove the hedge. Never include AI training disclaimers.

---

### 20. Sycophantic/Servile Tone

**Before:**
> Great question! You're absolutely right that this is complex. That's an excellent point about scalability.

**After:**
> Scalability is the hardest part of this architecture.

**Fix:** Skip the compliments. Get to the substance.

## Filler and Hedging

### 21. Filler Phrases

| Before | After |
|--------|-------|
| In order to achieve this goal | To achieve this |
| Due to the fact that it failed | Because it failed |
| At this point in time | Now |
| In the event that you need help | If you need help |
| The system has the ability to process | The system can process |
| It is important to note that the data shows | The data shows |

---

### 22. Excessive Hedging

**Before:**
> It could potentially possibly be argued that the approach might have some effect on outcomes.

**After:**
> The approach may affect outcomes.

**Fix:** One qualifier is enough. Two is suspicious. Three is AI.

---

### 23. Generic Positive Conclusions

**Before:**
> The future looks bright for the project. Exciting times lie ahead as they continue their journey toward excellence.

**After:**
> The team plans to ship v2.0 in March with the long-requested dark mode.

**Fix:** Replace vague optimism with specific next steps or concrete plans.

---

### 24. Performative Authenticity

**Watch for:** This is the part I keep coming back to, I had to look up most of these terms, What gets to me here is, The thing that surprised me was, As I was researching this

**Before:**
> This is the part I keep coming back to. I had to look up most of these terms before I could write this down.

**After:**
> [State the observation or fact directly. If the reaction is real and tied to specific content, ground it there ("the FreeBSD NFS exploit is the part that's hardest to dismiss"). Otherwise omit.]

**Fix:** Meta-commentary about the act of writing or researching is AI mimicking the texture of human reflection. Real writers either state observations directly or tie reactions to specific content. Cut the "I am being authentic now" flags.

---

### 25. Uniform Declarative Cadence

**Watch for:** Multiple short declarative sentences in sequence with similar length and subject-verb-object structure, each cleanly separated, no subordinate clauses

**Before:**
> The model found 22 bugs. Mozilla rated 14 as high severity. They shipped fixes. That was a fifth of all bugs. The team continued.

**After:**
> The model found 22 bugs over two weeks, 14 of which Mozilla rated high severity and shipped fixes for. That was about a fifth of all the high-severity bugs Mozilla fixed that year.

**Fix:** Real prose mixes rhythms. Combine some sentences with conjunctions or subordinate clauses to create flow. Each individual short sentence can be "clean," but stacked together they read mechanical. Humans vary length and structure.

## More Tells

### 26. Fabricated Names, Titles, and Quotes

**Watch for:** recurring filler names (Emily Carter, Sarah Thompson, John Smith), everyone sharing one title (all "Dr." or "industry expert"), and quotes that sound like the surrounding article

**Before:**
> Dr. Emily Carter, a leading expert, said: "This represents a paradigm shift that will fundamentally transform the industry."

**After:**
> Maria Okonkwo, who runs the maintenance crew, put it plainly: "We were duct-taping it every Friday."

**Fix:** Use specific, varied, contextual names and titles. Real quotes are shorter, rougher, and don't mirror the prose around them. If a person or quote is invented to fill space, cut it.

---

### 27. Staccato Fragment Triads

**Watch for:** "No X. No Y. Just Z.", "Focused. Aligned. Measurable.", one-word sentences stacked for punch (a LinkedIn/marketing tell)

**Before:**
> No fluff. No filler. Just results. Fast. Reliable. Scalable.

**After:**
> It ships in two weeks and handles 10k requests a second without falling over.

**Fix:** Replace the slogan rhythm with one real sentence that carries actual information.

---

### 28. Question-as-Transition

**Watch for:** staged setup-and-answer used as a connector: "The result? ...", "Why does this matter? ...", "The catch? ..."

**Before:**
> We rewrote the parser. The result? A 3x speedup.

**After:**
> We rewrote the parser and it runs 3x faster.

**Fix:** Drop the rhetorical question and state the result. One or two genuine questions in a piece are fine; a string of them is a tell.

---

### 29. Opener Clichés

**Watch for:** "In today's fast-paced world," "In the ever-evolving landscape of," "Picture this:", "Ever wondered...", "Here's the kicker/deal," "Let's dive in"

**Before:**
> In today's fast-paced digital world, businesses must constantly innovate to stay ahead.

**After:**
> Last quarter we lost two customers to a competitor who shipped the feature we'd been sitting on.

**Fix:** Open with something specific: a fact, a moment, a stake. Cut the throat-clearing windup entirely.

## Add voice

Removing patterns leaves the writing clean but empty. After scrubbing, put a person back in:

1. Add one real detail from experience. "In our case...", a specific number, a specific failure.
2. Take a position. "Honestly, I think this is overrated." An opinion beats balanced mush.
3. Break the rhythm. A long sentence, then a short one. A fragment, on purpose.
4. Admit a limit. "I haven't tested X." Perfect coverage reads as AI.
