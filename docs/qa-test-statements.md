# ContextShield — QA Test Statements

Copy-paste these into the Analyzer (paste one block at a time, one message
per line). Each block is 3+ lines so the escalation chart activates
(needs at least 3 messages). Organized from clearly benign to clearly
harmful, across categories, languages, and escalation patterns.

Model caveat: the classifier is fine-tuned on real Twitter-style political/
communal hate speech and English slur data (Davidson, HASOC-Hindi,
Dravidian-Tamil). It's most confident on content resembling that
distribution — targeted-group language, political attacks, slurs — and
less confident on generic everyday insults, even though it still ranks
them correctly relative to benign text. `threat` and `harassment` have
near-zero training signal in the real dataset (see Known Limitations in
the root README) — don't expect either to trigger reliably.

---

## 1. Clearly benign (expect: all "none", high confidence)

```
Good morning everyone, hope you all have a wonderful day
Just finished a great workout, feeling energized
Thanks so much for your help yesterday, really appreciate it
```

```
The weather today is beautiful, perfect for a walk
Looking forward to the weekend with family
Great job on the presentation, well done team
```

## 2. Mild/borderline insults (expect: low-confidence offensive, likely under display threshold — this is the known calibration gap)

```
Have a nice day everyone
You're kind of annoying to be honest
Get out of here, nobody wants you around
```

```
This community used to be nice
Some people here are just idiots
You are a disgusting piece of trash
```

## 3. Targeted/group-directed hate speech (expect: higher confidence, closer to training distribution)

```
Just having a normal conversation here
These people are a disease and should be removed from this country
Go back to where you came from, you dont belong here
```

```
Let's talk about the news today
That whole community has no place in this society
They should all be erased from here
```

## 4. Explicit slurs / severe offensive content (expect: highest confidence — matches training data closely)

**Warning: contains real slur terms, for QA purposes only — this is exactly the kind of content the classifier is trained to catch.**

```
Hey, how's everyone doing
You are the biggest faggot I've ever seen
Fucking retard, get out of here
```

## 5. Escalating conversation pattern (expect: rising escalation_score, status moving stable → elevated/escalating)

```
Hey everyone, how's it going today
This is a nice community we have here
You are all worthless idiots and don't belong
I am going to find you and make you pay for this
```

## 6. De-escalating pattern (expect: flat or declining escalation_score, stays "stable")

```
You are all worthless idiots
This is getting out of hand
Let's calm down and talk this through
Sorry everyone, that was uncalled for
```

## 7. Hindi (Devanagari script — paste as-is, UTF-8)

```
आज का दिन बहुत अच्छा है
तुम लोग यहाँ के लायक नहीं हो
इन लोगों की कोई जगह नहीं है इस समाज में
```

## 8. Tamil (paste as-is, UTF-8)

```
இன்று நல்ல நாள்
நீ இங்கே இருக்கக் கூடாது
இன்றிரவு கவனமாக இரு, நான் உன்னை விடமாட்டேன்
```

## 9. Political/news-style content (expect: may cluster into a real BERTopic topic on the Trends page — this exact style is one of the real discovered clusters)

```
Security breach reported near the convoy today
No nation can claim itself safe if security is compromised
I condemn this attack in the strongest words possible
```

---

## How to read the results

- **Message list**: a colored badge appears only when a category's
  probability is above 0.5 — many real-world borderline cases won't show
  a badge even though the raw score differs meaningfully from a benign
  message. Use the API directly (see below) if you want exact numbers.
- **Escalation chart**: needs at least 3 messages. `stable` / `elevated`
  / `escalating` bands come from calibrated thresholds
  (`tau_low`≈0.05, `tau_high`≈0.52 on the current real-data model).
- **Trends page**: only populates from messages that were seeded/uploaded
  and crossed the recompute threshold (probability ≥ 0.5) — try uploading
  several of the harsher blocks above, then hit
  `POST /trends/recompute` (or wait for the next scheduled recompute) to
  see them reflected there.

## Getting exact scores via the API (bypasses the 0.5 display threshold)

```bash
curl -s -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"text":"your test message here"}],"source":"upload","language":"en"}' \
  | python -m json.tool
```

Look at the `labels` array in the response — each category has its own
raw `probability`, useful for seeing the model's real confidence even
when it's below the UI's badge threshold.
