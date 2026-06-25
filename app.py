# app.py — Student Mental Health Risk Classifier
# Run: python app.py

import pickle, os, heapq, numpy as np
from colorama import Fore, Style, init
from tracker import log_today, weekly_summary
init(autoreset=True)

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load Model ────────────────────────────
try:
    with open(os.path.join(BASE,'mental_health_model.pkl'),'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE,'scaler.pkl'),'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE,'feature_cols.pkl'),'rb') as f:
        feature_cols = pickle.load(f)
    print(Fore.GREEN + "✅ Model loaded successfully!")
except FileNotFoundError as e:
    print(Fore.RED + f"\n❌ File not found: {e}")
    print("Make sure all 3 pkl files are in the same folder.\n")
    exit()

# ── A* Recovery Interventions ─────────────
INTERVENTIONS = [
    ('🌙 Fix sleep: 8 hrs, same time daily',        1, 3),
    ('🏃 30-min walk or light exercise daily',      1, 2),
    ('📵 No phone 1 hr before sleeping',            1, 2),
    ('🧘 5-min deep breathing when stressed',       1, 2),
    ('📖 Write 3 good things in journal daily',     1, 2),
    ('👥 Talk to one trusted friend this week',     2, 3),
    ('🎵 Calming music for 15 min/day',             1, 1),
    ('☕ No caffeine after 2pm',                    1, 1),
    ('🥗 Eat 3 proper meals — no skipping',         1, 2),
    ('📅 Make a weekly study schedule',             2, 2),
    ('🏫 Visit university counseling center',       2, 4),
    ('💊 Consult a professional therapist',         4, 5),
    ('🎨 30 min on a hobby you love',               1, 2),
    ('🚶 10-min break every study hour',            1, 1),
    ('🌿 Go outside in sunlight for 15 min',        1, 2),
]

def heuristic(s): return s // 2

def astar(start, goal=0):
    queue   = [(heuristic(start), 0, start, [])]
    visited = set()
    while queue:
        f, g, state, path = heapq.heappop(queue)
        if state in visited: continue
        visited.add(state)
        if state <= goal: return path, g
        for name, cost, imp in INTERVENTIONS:
            ns = max(0, state - imp)
            heapq.heappush(queue,
                (g+cost+heuristic(ns), g+cost, ns, path+[name]))
    return path, g

RISK_STATE = {0: 2, 1: 6, 2: 10}

# ── Input Helpers ─────────────────────────
def ask_choice(prompt, options):
    while True:
        print(prompt)
        for k, v in options.items():
            print(f"  {k} → {v}")
        val = input(Fore.CYAN+"  Your answer: "+Style.RESET_ALL).strip()
        if val in options: return val
        print(Fore.RED+f"  ⚠ Enter one of: {list(options.keys())}\n")

def ask_float(prompt, mn, mx):
    while True:
        val = input(Fore.CYAN+f"  {prompt} ({mn}–{mx}): "+Style.RESET_ALL).strip()
        try:
            n = float(val)
            if mn <= n <= mx: return n
            print(Fore.RED+f"  ⚠ {mn} aur {mx} ke beech likho\n")
        except: print(Fore.RED+"  ⚠ Valid number likho\n")

def ask_int(prompt, mn, mx):
    while True:
        val = input(Fore.CYAN+f"  {prompt} ({mn}–{mx}): "+Style.RESET_ALL).strip()
        try:
            n = int(val)
            if mn <= n <= mx: return n
            print(Fore.RED+f"  ⚠ {mn} aur {mx} ke beech likho\n")
        except: print(Fore.RED+"  ⚠ Poora number likho\n")

# ── Main App ──────────────────────────────
def run():
    print("\n" + "="*55)
    print(Fore.CYAN+Style.BRIGHT+
          "   🎓 Student Mental Health Risk Classifier")
    print(Fore.CYAN+
          "      ML + Genetic Algorithm + A* Search")
    print("="*55+"\n")
    print("Sawaalon ke jawab do — sirf 1 minute lagega.\n")

    print(Fore.YELLOW+"\n── Personal Information ─────────────────────")
    g      = ask_choice("\n1. Gender?", {'1':'Male','2':'Female'})
    gender = 1 if g=='1' else 0
    age    = ask_int("\n2. Age?", 15, 40)
    year   = ask_int("\n3. Year of study?", 1, 4)
    cgpa   = ask_float("\n4. CGPA?", 0.0, 4.0)
    m      = ask_choice("\n5. Married?", {'1':'Yes','2':'No'})
    married = 1 if m=='1' else 0
    sleep  = ask_float("\n6. Average sleep hours per night?", 1.0, 12.0)
    sp     = ask_int("\n7. Social support? (1=None  5=Great)", 1, 5)
    ap     = ask_int("\n8. Academic pressure? (1=Low  5=High)", 1, 5)
    sh     = ask_int("\n9. Daily study hours?", 1, 12)
    fs     = ask_int("\n10. Financial stress? (1=None  5=Severe)", 1, 5)
    pa     = ask_int("\n11. Exercise days per week?", 0, 7)

    print(Fore.YELLOW+"\n── Mental Health Symptoms ───────────────────")
    d = ask_choice("\n12. Feel depressed? (sadness, hopelessness)?",
                   {'1':'Yes','2':'No'})
    depression = 1 if d=='1' else 0

    a = ask_choice("\n13. Experience anxiety? (worry, nervousness)?",
                   {'1':'Yes','2':'No'})
    anxiety = 1 if a=='1' else 0

    p = ask_choice("\n14. Have panic attacks?", {'1':'Yes','2':'No'})
    panic = 1 if p=='1' else 0

    t = ask_choice("\n15. Sought specialist treatment?", {'1':'Yes','2':'No'})
    treatment = 1 if t=='1' else 0

    # Build input
    input_dict = {
        'Gender': gender, 'Age': age, 'Year': year,
        'CGPA': cgpa, 'Married': married, 'SleepHours': sleep,
        'SocialSupport': sp, 'AcademicPressure': ap,
        'StudyHours': sh, 'FinancialStress': fs,
        'PhysicalActivity': pa, 'Depression': depression,
        'Anxiety': anxiety, 'PanicAttack': panic, 'Treatment': treatment
    }

    input_arr    = np.array([[input_dict[c] for c in feature_cols]])
    input_scaled = scaler.transform(input_arr)
    prediction   = model.predict(input_scaled)[0]
    proba        = model.predict_proba(input_scaled)[0]

    risk_labels = {0:'Low Risk', 1:'Medium Risk', 2:'High Risk'}
    risk_colors = {0:Fore.GREEN, 1:Fore.YELLOW,  2:Fore.RED}
    risk_label  = risk_labels[prediction]
    risk_color  = risk_colors[prediction]

    # Results
    print("\n\n"+"="*55)
    print(Fore.CYAN+Style.BRIGHT+"   📊 RESULTS")
    print("="*55)
    print(f"\n  Predicted Risk  : "+risk_color+Style.BRIGHT+risk_label)
    print(Style.RESET_ALL+f"  Confidence      : {proba[prediction]*100:.1f}%")
    print(f"\n  Probability Breakdown:")
    print(f"    🟢 Low Risk    : {proba[0]*100:.1f}%")
    print(f"    🟡 Medium Risk : {proba[1]*100:.1f}%")
    print(f"    🔴 High Risk   : {proba[2]*100:.1f}%")

    # Insight
    print("\n"+"─"*55)
    print(Fore.CYAN+"  What this means:")
    print(Style.RESET_ALL)
    if prediction == 0:
        print("  ✅ Aap ki mental health achi hai!")
        print("  Apna healthy routine jaari rakho.")
        print("  Dosto ki bhi madad karo jo struggle kar rahe hain.")
    elif prediction == 1:
        print("  ⚠️  Aap moderate stress mein hain.")
        print("  Abhi action lena zaroori hai —")
        print("  choti choti improvements bohat farq karti hain.")
    else:
        print("  🚨 Significant mental health challenges hain.")
        print("  Help maangna kamzori nahi — strength ki nishani hai.")
        print("  Aap akele nahi hain. 💙")

    # Recovery Plan
    print("\n"+"─"*55)
    print(Fore.CYAN+"  🗺️  Personalized Recovery Plan (A* Optimized):")
    print(Style.RESET_ALL)

    path, cost = astar(RISK_STATE[prediction])

    if not path:
        print("  ✅ Aap already healthy state mein hain!")
    else:
        for i, step in enumerate(path, 1):
            print(f"  Step {i}: {step}")

    print(f"\n  Efficiency Score: {cost} (kam = zyada efficient)")

    if prediction == 2:
        print("\n"+Fore.RED+"─"*55)
        print(Fore.RED+Style.BRIGHT+"  ⚠  Crisis mein ho toh:")
        print(Fore.RED+"     University counseling center zaroor jao.")
        print(Fore.RED+"     Ya kisi trusted insaan se baat karo.")
        print(Style.RESET_ALL)

    print("="*55+"\n")

    # Tracker
    name = input("Apna naam batao (tracker ke liye, Enter skip): ").strip()
    if name:
        log_today(name, risk_label)
        show = input("\nWeekly summary dekhni hai? (y/n): ").strip().lower()
        if show == 'y':
            weekly_summary(name)

    again = input("\nDobara chalaana hai kisi aur student ke liye? (y/n): ").strip().lower()
    if again == 'y':
        run()
    else:
        print(Fore.CYAN+"\nShukriya. Apna khayal rakhna. 💙\n")

if __name__ == "__main__":
    run()
