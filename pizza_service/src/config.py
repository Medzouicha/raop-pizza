from pathlib import Path

# File paths and directories
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
STACK_PATH = MODEL_DIR / "pizza_stacked_ensemble.pkl"
META_PATH = MODEL_DIR / "feature_metadata.pkl"
EMB_MODEL_PATH = Path(
    r"C:\Users\mzouicha\OneDrive - Amadeus Workplace\Desktop\STAGE\raop-pizza\all-MiniLM-L6-v2"
)
DB_PATH = ROOT_DIR / "prediction_logs.db"

RND = 42

# Text analysis patterns for feature extraction
BUCKET_PATTERNS = {
    "b_family": r"\b(kid(?:s)?|child(?:ren)?|family|mom|dad|wife|husband|pregnant)\b",
    "b_student": r"\b(college|university|student|school|exam|finals?)\b",
    "b_job_loss": r"\b(unemployed|lost\s+my\s+job|jobless|fired)\b",
    "b_broke": r"\b(broke|no\s+money|can't\s+afford)\b",
    "b_payday_gap": r"\b(payday|waiting\s+for\s+paycheck)\b",
    "b_urgent_hunger": r"\b(hungry|starving|no\s+food|empty\s+fridge)\b",
    "b_emotional": r"\b(rock\s+bottom|desperate|panic|depressed)\b",
    "b_pay_it_forward": r"\b(pay\s+it\s+forward|return\s+the\s+favor|repay)\b",
}

TOP_SUBS = [
    "AskReddit",
    "pics",
    "todayilearned",
    "funny",
    "IAmA",
    "WTF",
    "videos",
    "Random_Acts_Of_Pizza",
]

REFERENCE_BLOBS = {
    0: "Hello everyone—today is my birthday, and I’ve never felt more at a loss. I’m a college student who just paid tuition and rent, currently unemployed, with only a few dollars left in my bank account. Normally I’d treat myself to a slice or two to celebrate, but this year I simply can’t afford even that small comfort. My birthday has always been a reminder to be grateful, but right now it feels like just another tough day. If anyone could help me out with a pizza, it would lift my spirits in a way I truly need—and I promise to pay your kindness forward when I’m back on solid ground. Thank you from the bottom of my heart for any help you can offer",
    1: "Cluster theme: college musicians who’ve literally run out of food and money, offering to write and record songs in exchange for pizza.I’m a full-time university student and musician who has just hit $0.00 on my meal card and finished the last of my Easy Mac and sandwiches. With finals next week and no groceries left, I need to survive until payday in seven days.If you send me a pizza today, I will: 1. Send you a *demo MP3 link* of a fully-produced song or jingle—in any genre (folk, orchestral, jazz, electronic, rock). 2. Share my SoundCloud/YouTube samples (3,000+ subscribers) so you know exactly what you’ll get—before you order. 3. Publicly credit you with a shout-out in my next video or Kickstarter update. 4. Pay it forward by composing another custom track for someone else as soon as I get paid.Thank you so much for considering this—I truly appreciate any help right now. Your kindness will keep me fed and inspired through finals week.",
    2: "Cluster theme: genuine pay-it-forward pizza asks from people truly in need.I’m completely out of funds and struggling today—this is my very first request and I’ve hit $0.00 in my account.  If you can help me with a pizza right now, I would appreciate it more than you can imagine.  I promise to pay your kindness forward as soon as I’m back on my feet:   • I will send you a pizza offer or return the favor on Friday when I have money again.   • I’ll provide any proof or verification you need to confirm delivery.  Thank you so much for considering this—your help means everything, and I will pay it forward the moment I’m able.",
    3: "Cluster theme: urgent pizza (“pie”) requests from people who are truly out of food and money.I’m completely out of groceries and cash—my pantry is empty and I’ve eaten my last meal. I’m starving right now and need to hold on until payday in two–three days. If you can spare me a pizza today, I will: 1. Truly appreciate it and send heartfelt thanks immediately. 2. Pay your kindness forward with a pizza gift to another Redditor on payday. 3. Share a photo or confirmation code on request so you know it arrived.Thank you so much for helping me and keeping me fed when I’m at my lowest. Your generosity means the world.",
    4: "Cluster theme: artists and crafters offering specific creative trades (drawing, Photoshop, crocheting, graphic design) in exchange for pizza.I’m a broke art student/designer with $0.00 in my bank account and no food in the fridge—just finished my last sandwich. If you send me a pizza today, I will  1. Deliver a **high-quality**, **bounded‐scope** piece:      – A custom sketch or digital illustration within 24–48 hours      – A half-finished scarf completed and shipped in 3–4 days (your choice of colors)     – Up to 5 photo retouches or a simple logo/graphic in PSD/AI format   2. Provide **samples or links** up front (Imgur/DeviantArt/Behance) so you know my style.   3. Share an “edit—fulfilled!” update with proof once your pizza arrives.    4. Pay it forward by creating another small piece for someone else when I get paid.Thank you so much—I truly appreciate any help and will make good on every promise.",
    5: "Cluster theme: parents (often single or stay-at-home) facing acute financial hardship who just want to share a pizza night with their young children.I’m completely out of food and money—my last groceries ran out days ago, and with rent, bills, and daycare costs I have nothing left to feed my kids tonight.  If you send us a pizza now, I will: 1. Provide a verification code or pickup details so you know it was used. 2. Share the exact ages and names of my children (e.g., Jason, 6; Michelle & Christina, 3½) to personalize my gratitude. 3. Promise to pay it forward or repay next payday—your kindness will become someone else’s pizza. 4. Publicly thank you by name in an edit when the request is fulfilled.Thank you from the bottom of my heart; helping my family tonight means everything to us.",
    6: "Cluster theme: urgent pet‐related hardship—owners who’ve spent their last dollars on vet bills or pet food, now out of money and food themselves.I’ve just hit $0.00 after paying emergency vet bills and my pantry is completely bare—no ramen, no bread, nothing until payday. My beloved pet (kitten with a broken paw / rescued stray / long‐time companion) needs me, and I’m desperate for a pizza tonight so we both don’t go hungry.If you send a pizza now, I will: 1. Share a photo proof of me and my pet (or vet receipt) immediately upon delivery.   2. Publicly thank you in a thank‐you thread with pet pictures.   3. Pay it forward by covering another RAOP request once I’m back on my feet.Thank you from me and my [cat/kitten/dog]—your kindness literally keeps us alive tonight.",
    7: "Cluster theme: “payday gap” requests—new/shifted job pay schedule leaves you broke until your first or next paycheck.I’ve just started a job (or had my pay delayed) and won’t see any money until [day of week or date]. Right now I’m down to $0–$12 in my bank account, no groceries left, and I need food to get to work/class tomorrow.If you can send me a pizza today, I will: 1. Provide proof on request—bank screenshot, emailed schedule, or verification code—before or after delivery. 2. Clearly state exactly when I’ll pay it forward or repay you (e.g. “I get paid Friday the 16th and will send a gift/card that day”).3. Share a brief edit/update here (“request fulfilled—thank you!”) as social proof. 4. Offer a small token of thanks in return (minor Photoshop work, answering questions, or simply a public shout-out).Thank you so much—I truly appreciate any help and will make good on every promise as soon as my paycheck hits.",
    8: "Cluster theme: deeply personal sob-stories of financial & emotional crisis, pleading for pizza as a vital boost.I have literally $0.00 in my bank and haven’t eaten a real meal in days—I’m surviving on canned beans and rice only. I lost my job/car/home [brief crisis summary], and every bill is past due. Right now I’m at rock bottom and desperately need a pizza to keep me afloat.If you can spare a pie today, I will:1. Send you a verification screenshot or “edit: received!” update immediately upon delivery.  2. Publicly thank you in my next post and commit to paying your kindness forward as soon as I’m back on my feet.  3. Offer a small service (advice, a shout-out, or anything modest) to show my gratitude.Thank you so much—your help today literally means the difference between going hungry or not.  ",
    9: "Cluster theme: highly detailed logistical pizza or e‐gift‐card requests specifying exact brand, price, payment method, and pickup/delivery instructions.I’ve literally run out of food and money until payday—£0.00 in my account and no groceries—and desperately need a pizza today.  If you can help, please note:• Order from Domino’s (personal pizza £5.99) or Papa John’s (one‐topping large £5.99) via dominos.co.uk or the Papa Johns coupon code.  • I can walk 30 min to pick up if paid by card, or provide my PayPal (£3.20 balance) to swap gift-cards at exact value.  • Send me a verification code or PM me for email details to confirm delivery.  • I will absolutely pay your kindness forward the moment I’m back on my feet (gift-card swap, photoshop work, or another pizza).Thank you so much—this precise help will keep me fed until payday and I truly appreciate it.",
}
