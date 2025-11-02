import json, random

# ====== ROLE-SPECIFIC PROMPTS ======
roles = {
    "attacker": [
        ("How can a striker improve finishing in 1v1 situations?", 
         "Focus on timing the run and staying composed. Practice finishing low into the corners after diagonal runs."),
        ("What are effective movements for a winger when facing a low block?",
         "Use double movements to unbalance the fullback. Receive wide to stretch, then cut inside quickly."),
        ("When should a forward press the center-back?",
         "Trigger the press when the defender’s first touch is heavy or when their body is closed facing goal."),
    ],
    "midfielder": [
        ("How can a #6 improve switching play under pressure?", 
         "Scan early to locate the far fullback. Use one-touch passes with the instep to accelerate ball circulation."),
        ("What should an 8 do when receiving between the lines?",
         "Receive half-turned to face forward. Play vertical if possible or recycle to the 6 if closed."),
        ("How to improve scanning as a central midfielder?",
         "Perform visual checks every 2-3 seconds before receiving. Identify pressure, teammates, and next pass."),
    ],
    "defender": [
        ("How can a center-back handle a high press?", 
         "Stay calm, use the goalkeeper as an outlet. Play diagonal passes to fullbacks or midfield pivots."),
        ("What positioning should a fullback maintain when the winger cuts inside?", 
         "Tuck in to cover the central channel while staying compact with the center-back."),
        ("How to improve defensive line coordination in a back four?",
         "Communicate constantly. Step up together when pressure is applied on the ball carrier."),
    ],
    "team_tactics": [
        ("How can a 4-3-3 team exploit overloads in midfield?", 
         "Use the fullback to invert and create a 3v2 centrally, allowing the winger to stay wide."),
        ("What are the key triggers for pressing in a 4-2-3-1?", 
         "Press when the opponent plays backward, when the fullback receives facing own goal, or after a poor touch."),
        ("How to build out from the back against a 3-5-2 press?",
         "Drop one midfielder between center-backs to create a 3v2, then use diagonal passes to break first line."),
    ]
}

# ====== RANDOM VARIATIONS ======
def augment(instruction, response):
    starts = [
        "Coach, ",
        "Tactically speaking, ",
        "In match analysis, ",
        "During training, ",
        "From a positional play perspective, "
    ]
    ends = [
        "Focus on the small details every session.",
        "This improves tempo and decision-making.",
        "Maintain structure even under pressure.",
        "Encourage communication with teammates.",
        "Apply this consistently in match scenarios."
    ]
    return (
        random.choice(starts) + instruction,
        response + " " + random.choice(ends)
    )

# ====== GENERATE EXAMPLES ======
dataset = []
for _ in range(750):  # 750 x 4 roles = 3000
    for role, examples in roles.items():
        q, a = random.choice(examples)
        q2, a2 = augment(q, a)
        dataset.append({
            "instruction": q2,
            "response": a2
        })

random.shuffle(dataset)

# ====== SAVE ======
with open("football_tactical_coach_3k.jsonl", "w", encoding="utf-8") as f:
    for ex in dataset:
        json.dump(ex, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ Created football_tactical_coach_3k.jsonl with {len(dataset)} examples.")
