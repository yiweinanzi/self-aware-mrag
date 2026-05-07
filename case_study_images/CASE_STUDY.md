# OK-VQA Case Study

## Case 1

**Image:** `case1.jpg` ![case1.jpg](case1.jpg)

**Question:**
What sport can you use this for?

**Gold Answer:**
race, race, race

**Baseline Output (ViDoRAG):**
None of the listed sports (squash, roller hockey, lacrosse, rowing, racket sports) are directly related to

**Ours Output (Self-Aware-MRAG):**
motorcycle riding

**Top-3 Evidence Snippets:**

| DocID | Title | Snippet | Pos |
|-------|-------|---------|-----|
| 636978 | Squash (sport) | Squash (sport) is appropriate for one's skill level. The four different ball types are Intro (Blue dot, 140% of Pro bounce), Progress (Red dot, 120% of Pro bounce), Competition (single yellow dot, 110% of Pro bounce) and Pro (double yellow dot). | front |
| 352754 | Roller hockey | Roller hockey football or basketball. The stick is more or less the same as in bandy and shinty. Roller hockey (inline) bears close resemblance to ice hockey and is played on inline skates, uses an ice hockey stick and includes a lot of fast "racing back and forth" action. | mid |
| 248638 | Lacrosse | Lacrosse protect against concussions. Women goalies wear a helmet, gloves, and chest protector. Lacrosse Lacrosse is a team sport played with a lacrosse stick and a lacrosse ball, commonly played indoors or outdoors. | mid |

**Meta:**
- uncertainty_total=0.3131
- uncertainty_text=0.3900
- uncertainty_visual=0.3271
- retrieved=True
- pos_aware_on=True

---

## Case 2

**Image:** `case2.jpg` ![case2.jpg](case2.jpg)

**Question:**
Name the type of plant this is?

**Gold Answer:**
vine, vine, vine

**Baseline Output (ViDoRAG):**
Based on the visual evidence, the plant is a vertical garden or living wall composed of various green leafy plants, likely including

**Ours Output (Self-Aware-MRAG):**
houseplant

**Top-3 Evidence Snippets:**

| DocID | Title | Snippet | Pos |
|-------|-------|---------|-----|
| 1441102 | Asclepias tuberosa | Asclepias tuberosa Common names include butterfly weed, Canada root, chieger flower, chiggerflower, fluxroot, Indian paintbrush, Indian posy, orange milkweed, orange root, orange Swallow-wort, pleurisy root, silky swallow-wort, tuber root, yellow milkweed. | front |
| 2820794 | Aegopodium podagraria | Aegopodium podagraria Aegopodium podagraria (commonly called ground elder, herb gerard, bishop's weed, goutweed, gout wort, and snow-in-the-mountain, and sometimes called English masterwort and wild masterwort) is a perennial plant in the carrot family (Apiaceae). | mid |
| 2483675 | Pachypodium lamerei | Pachypodium lamerei Pachypodium lamerei is a species of flowering plant in the dogbane family Apocynaceae. It is a stem succulent, photosynthesizing mainly through its trunk, and comes from the island of Madagascar. | mid |

**Meta:**
- uncertainty_total=0.3470
- uncertainty_text=0.3900
- uncertainty_visual=0.4401
- retrieved=True
- pos_aware_on=True

---

## Case 3

**Image:** `case3.jpg` ![case3.jpg](case3.jpg)

**Question:**
What toy is this?

**Gold Answer:**
stuffed animal, stuffed animal, stuffed animal

**Baseline Output (ViDoRAG):**
stuffed toy

**Ours Output (Self-Aware-MRAG):**
teddy bear

**Top-3 Evidence Snippets:**

| DocID | Title | Snippet | Pos |
|-------|-------|---------|-----|
| 990298 | Yo-yo | Yo-yo A yo-yo (also spelled yoyo) is a toy consisting of an axle connected to two disks, and a string looped around the axle. It has some similarity to a yo-yo in terms of up and down motion. | front |
| 1327296 | Toy | Toy A toy is an item that is used in play, especially one designed for such use. Playing with toys can be an enjoyable means of training young children for life in society. | mid |
| 2349108 | Stuffed toy | Stuffed toy A stuffed toy is a toy with an outer fabric sewn from a textile and then stuffed with a flexible material. In North American English, they are variously referred to as plush toys, plushies, stuffed animals. | mid |

**Meta:**
- uncertainty_total=0.3053
- uncertainty_text=0.3500
- uncertainty_visual=0.3675
- retrieved=True
- pos_aware_on=True

---

## Case 4

**Image:** `case4.jpg` ![case4.jpg](case4.jpg)

**Question:**
Which part of this animal would be in use of it was playing the game that is played with the items the man is holding?

**Gold Answer:**
mouth, mouth, mouth

**Baseline Output (ViDoRAG):**
paws

**Ours Output (Self-Aware-MRAG):**
paws and mouth

**Top-3 Evidence Snippets:**

| DocID | Title | Snippet | Pos |
|-------|-------|---------|-----|
| 1538472 | Learning | Learning in itself, but that improves performance in similar future situations. This is seen in a wide variety of vertebrates besides humans, but is most evident in humans. | front |
| 393652 | Hunting | Hunting Hunting is the human practice of seeking, pursuing, capturing, or killing wildlife or feral animals. | mid |
| 518823 | Instinct | Instinct Instinct is the inherent inclination of a living organism towards a particular complex behaviour. | mid |

**Meta:**
- uncertainty_total=0.3678
- uncertainty_text=0.4500
- uncertainty_visual=0.4095
- retrieved=True
- pos_aware_on=True

---

## Case 5

**Image:** `case5.jpg` ![case5.jpg](case5.jpg)

**Question:**
What could this gentleman be carrying in that red bag?

**Gold Answer:**
cloth, cloth, cloth

**Baseline Output (ViDoRAG):**
diplomatic bag

**Ours Output (Self-Aware-MRAG):**
diplomatic bag

**Top-3 Evidence Snippets:**

| DocID | Title | Snippet | Pos |
|-------|-------|---------|-----|
| 2938753 | Diplomatic bag | Diplomatic bag exactly this purpose. Diplomatic bag A diplomatic bag, also known as a diplomatic pouch, is a container with certain legal protections. | front |
| 4184543 | Luggage | Luggage Luggage consists of bags, cases and containers which hold a traveller's articles during transit. | mid |
| 454486 | Suitcase | Suitcase A suitcase is a form of luggage. | mid |

**Meta:**
- uncertainty_total=0.3628
- uncertainty_text=0.4500
- uncertainty_visual=0.3925
- retrieved=True
- pos_aware_on=True

---

## Summary

| Case | Question | Gold | Baseline | Ours | Correct |
|------|----------|------|----------|------|---------|
| 1 | What sport... | race | None of the listed... | motorcycle riding | ✗ |
| 2 | Name the type of plant | vine | vertical garden... | houseplant | ✗ |
| 3 | What toy is this | stuffed animal | stuffed toy | teddy bear | ~ |
| 4 | Which part of this animal | mouth | paws | paws and mouth | ~ |
| 5 | What in red bag | cloth | diplomatic bag | diplomatic bag | ✗ |

**Note:** Case 3 and 4 show partial correctness (teddy bear is a stuffed animal, paws and mouth includes the correct answer).
