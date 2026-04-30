"""
prevention_tips.py
------------------
Usage:
    from utils.prevention_tips import PREVENTION_TIPS, format_tips

    tips_text = format_tips("Tomato___Late_blight")
    print(tips_text)
"""

from typing import Dict, Any, Optional

# ---------------------------------------------------------------------------
# Master knowledge base  (key == ImageFolder class name)
# ---------------------------------------------------------------------------
PREVENTION_TIPS: Dict[str, Dict[str, Any]] = {

    # ── APPLE ────────────────────────────────────────────────────────────────
    "Apple___Apple_scab": {
        "disease": "Apple Scab",
        "pathogen": "Venturia inaequalis (fungus)",
        "symptoms": (
            "Olive-green to brown velvety spots on leaves and fruit; "
            "infected leaves may curl and drop early; fruit shows dark, "
            "corky, scab-like lesions."
        ),
        "causes": (
            "Thrives in cool, wet spring weather (10–24 °C). "
            "Spores overwinter in fallen infected leaves and spread via rain splash."
        ),
        "prevention": [
            "Plant scab-resistant apple varieties (e.g., Liberty, Enterprise).",
            "Rake and destroy fallen leaves every autumn to eliminate overwintering spores.",
            "Prune to improve canopy airflow and reduce leaf wetness periods.",
            "Apply preventive fungicide sprays starting at green-tip stage.",
            "Avoid overhead irrigation; use drip systems instead.",
        ],
        "treatment": [
            "Apply fungicides (captan, myclobutanil, or copper-based) at 7–10 day intervals during wet periods.",
            "Begin treatments early in the season before infection periods.",
            "Remove and bag heavily infected fruits and leaves.",
            "Follow a complete spray calendar through primary scab season (spring).",
        ],
    },

    "Apple___Black_rot": {
        "disease": "Apple Black Rot",
        "pathogen": "Botryosphaeria obtusa (fungus)",
        "symptoms": (
            "Circular, purple-bordered brown spots on leaves ('frog-eye' lesions); "
            "fruit shows brown rot starting at the calyx end, eventually turning black "
            "and mummified; cankers on branches."
        ),
        "causes": (
            "Fungal spores spread from mummified fruit and dead bark. "
            "Warm, humid conditions (24–29 °C) with prolonged leaf wetness favor infection."
        ),
        "prevention": [
            "Remove and destroy mummified fruits and dead/cankered wood.",
            "Prune out diseased branches at least 15 cm below visible infection.",
            "Maintain good orchard sanitation; remove all debris after harvest.",
            "Avoid wounding trees; protect pruning cuts with wound sealant.",
        ],
        "treatment": [
            "Apply fungicides (thiophanate-methyl, captan) starting at petal fall.",
            "Repeat at 10–14 day intervals during summer.",
            "Surgically remove cankers on larger limbs and treat with fungicide.",
        ],
    },

    "Apple___Cedar_apple_rust": {
        "disease": "Cedar Apple Rust",
        "pathogen": "Gymnosporangium juniperi-virginianae (fungus)",
        "symptoms": (
            "Bright orange-yellow spots on upper leaf surface in spring; "
            "tube-like spore structures on leaf undersides; "
            "fruit develops similar orange lesions and may drop prematurely."
        ),
        "causes": (
            "Requires two hosts to complete its life cycle: eastern red cedar/juniper "
            "and apple. Spores spread by wind in spring during wet weather."
        ),
        "prevention": [
            "Plant rust-resistant apple varieties (e.g., Redfree, William's Pride).",
            "Remove nearby eastern red cedar or juniper trees if feasible.",
            "Apply fungicides (myclobutanil, mancozeb) from pink bud through petal fall.",
            "Inspect cedars for orange gelatinous galls in spring and remove them.",
        ],
        "treatment": [
            "Fungicide applications (myclobutanil, propiconazole) at pink bud, full pink, petal fall, and 10 days later.",
            "Remove infected leaves and fruit to reduce secondary spread.",
        ],
    },

    "Apple___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Maintain regular pruning for good air circulation within the canopy.",
            "Apply balanced fertilization based on annual soil tests.",
            "Monitor regularly for early signs of pest or disease pressure.",
            "Use drip irrigation to keep foliage dry.",
            "Apply dormant oil spray each winter to control overwintering pests.",
        ],
        "treatment": [],
    },

    # ── BLUEBERRY ────────────────────────────────────────────────────────────
    "Blueberry___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Maintain soil pH between 4.5 and 5.5 for optimal growth.",
            "Mulch with pine bark or wood chips to conserve moisture and suppress weeds.",
            "Apply balanced, acidic fertilizer (ammonium sulfate) in spring.",
            "Prune annually to remove old, unproductive canes.",
            "Net plants during fruiting to protect from birds and reduce mechanical damage.",
        ],
        "treatment": [],
    },

    # ── CHERRY ───────────────────────────────────────────────────────────────
    "Cherry_(including_sour)___Powdery_mildew": {
        "disease": "Cherry Powdery Mildew",
        "pathogen": "Podosphaera clandestina (fungus)",
        "symptoms": (
            "White, powdery fungal coating on young leaves, shoots, and fruit; "
            "infected leaves may curl, distort, or drop; fruit can become russeted."
        ),
        "causes": (
            "Favored by warm days (20–27 °C), cool nights, and high humidity. "
            "Unlike most fungi, does NOT require free water for infection."
        ),
        "prevention": [
            "Plant mildew-resistant cherry varieties where available.",
            "Prune to open the canopy and increase air circulation.",
            "Avoid excessive nitrogen fertilization which promotes succulent growth.",
            "Apply preventive fungicide sprays before symptoms appear.",
        ],
        "treatment": [
            "Apply sulfur-based or systemic fungicides (myclobutanil, trifloxystrobin).",
            "Begin at first sign of infection and repeat every 10–14 days.",
            "Remove and destroy heavily infected shoots.",
        ],
    },

    "Cherry_(including_sour)___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Prune annually after harvest to maintain open structure.",
            "Apply dormant copper spray each winter before bud break.",
            "Monitor for brown rot symptoms during humid periods.",
            "Use drip or furrow irrigation to keep foliage dry.",
        ],
        "treatment": [],
    },

    # ── CORN (MAIZE) ─────────────────────────────────────────────────────────
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "disease": "Gray Leaf Spot (Cercospora Leaf Spot)",
        "pathogen": "Cercospora zeae-maydis (fungus)",
        "symptoms": (
            "Rectangular, tan to gray lesions with parallel sides running between leaf veins; "
            "lesions may merge under severe infection, causing complete leaf blight."
        ),
        "causes": (
            "Thrives in warm (25–30 °C), humid, and cloudy conditions. "
            "Crop residue on the soil surface is the primary source of inoculum."
        ),
        "prevention": [
            "Plant resistant hybrids — the most effective management strategy.",
            "Rotate corn with non-host crops (soybean, wheat) for at least one year.",
            "Till or bury crop residue to reduce surface inoculum.",
            "Avoid dense planting; maintain adequate row spacing for airflow.",
        ],
        "treatment": [
            "Apply foliar fungicides (strobilurins, triazoles) at VT/R1 growth stage if disease is present.",
            "Scout fields regularly from V8 onwards in susceptible hybrids.",
        ],
    },

    "Corn_(maize)___Common_rust_": {
        "disease": "Common Rust of Corn",
        "pathogen": "Puccinia sorghi (fungus)",
        "symptoms": (
            "Small, circular to elongated, brick-red powdery pustules on both leaf surfaces; "
            "pustules rupture and release rust-colored spores; "
            "heavy infection causes yellowing and early senescence."
        ),
        "causes": (
            "Spores are wind-blown long distances. Favored by cool temperatures (16–23 °C) "
            "and high relative humidity or dew."
        ),
        "prevention": [
            "Use resistant or tolerant hybrid varieties.",
            "Early planting to avoid the peak spore dispersal period.",
            "Monitor crops regularly from the V6 stage onward.",
        ],
        "treatment": [
            "Fungicide application (propiconazole, azoxystrobin) is effective if applied early.",
            "Economic threshold: treat if rust is found on leaves below the ear before silking in susceptible hybrids.",
        ],
    },

    "Corn_(maize)___Northern_Leaf_Blight": {
        "disease": "Northern Corn Leaf Blight (NCLB)",
        "pathogen": "Exserohilum turcicum (fungus)",
        "symptoms": (
            "Long (2.5–15 cm), elliptical, cigar-shaped gray-green to tan lesions; "
            "lesions may show a 'dirty' appearance with dark fungal sporulation; "
            "severe infection causes significant yield loss."
        ),
        "causes": (
            "Cool (18–27 °C), moist conditions with extended leaf wetness periods. "
            "Survives in infected residue; spreads by wind and rain splash."
        ),
        "prevention": [
            "Plant NCLB-resistant hybrids (most economical control).",
            "Crop rotation to reduce residue inoculum.",
            "Tillage to bury infected residue.",
            "Avoid irrigating in the evening which prolongs leaf wetness.",
        ],
        "treatment": [
            "Apply fungicides (azoxystrobin + propiconazole) at V8–V10 if disease is detected.",
            "Efficacy is highest when applied before disease reaches the ear leaf.",
        ],
    },

    "Corn_(maize)___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Follow a proper crop rotation schedule.",
            "Test soil annually and apply balanced NPK fertilizer.",
            "Monitor for insect pests (earworm, rootworm) that can vector disease.",
            "Maintain optimal plant population density per variety recommendations.",
        ],
        "treatment": [],
    },

    # ── GRAPE ────────────────────────────────────────────────────────────────
    "Grape___Black_rot": {
        "disease": "Grape Black Rot",
        "pathogen": "Guignardia bidwellii (fungus)",
        "symptoms": (
            "Small, circular, tan to brown leaf spots with dark borders; "
            "infected berries turn brown, then black, shrivel into hard "
            "mummified 'raisins' that remain attached to the cluster."
        ),
        "causes": (
            "Overwinters in mummified berries and infected canes. "
            "Spores released during rain in spring; infection requires 36–48 hours of wetness at 10–32 °C."
        ),
        "prevention": [
            "Remove and destroy all mummified fruit and infected canes during winter pruning.",
            "Train vines to maximize sunlight penetration and air circulation.",
            "Apply protectant fungicides starting at bud break.",
            "Avoid planting in low-lying, poorly drained sites.",
        ],
        "treatment": [
            "Apply fungicides (myclobutanil, mancozeb, captan) every 10–14 days from bud break through veraison.",
            "Critical spray windows: at 2–3 inch shoot growth, at bloom, and 2–3 weeks post-bloom.",
            "Remove visibly infected clusters as soon as detected.",
        ],
    },

    "Grape___Esca_(Black_Measles)": {
        "disease": "Esca (Black Measles / Grapevine Trunk Disease)",
        "pathogen": "Phaeomoniella chlamydospora, Phaeoacremonium spp., Fomitiporia mediterranea (fungal complex)",
        "symptoms": (
            "Tiger-stripe pattern of yellow and brown interveinal leaf scorching; "
            "berries develop dark, sunken spots ('black measles'); "
            "internal wood shows brown streaking; chronic form causes slow vine decline."
        ),
        "causes": (
            "Fungi colonize woody tissue through pruning wounds; "
            "stress factors (drought, over-cropping) trigger symptom expression."
        ),
        "prevention": [
            "Apply wound protectants (Trichoderma-based or fungicide paint) immediately after pruning.",
            "Prune during dry weather and avoid large pruning wounds.",
            "Delay pruning until late in the dormant season to reduce wound exposure time.",
            "Remove and burn infected wood promptly.",
            "Maintain vine vigor through balanced irrigation and nutrition.",
        ],
        "treatment": [
            "No fully curative chemical treatment exists at present.",
            "Surgically remove infected wood ('curettage') to clean tissue.",
            "Re-train a healthy sucker shoot to replace heavily infected vines.",
            "Apply sodium arsenite (where legally permitted) as a wound treatment — limited efficacy.",
        ],
    },

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease": "Grape Leaf Blight (Isariopsis Leaf Spot)",
        "pathogen": "Pseudocercospora vitis (syn. Isariopsis clavispora) (fungus)",
        "symptoms": (
            "Irregular, dark brown to black spots on the upper leaf surface; "
            "spots may merge to produce large blighted areas; "
            "premature defoliation; berries can also be affected."
        ),
        "causes": (
            "Favored by warm (25–30 °C), humid conditions and prolonged leaf wetness. "
            "Spreads via wind-blown and rain-splashed conidia."
        ),
        "prevention": [
            "Ensure good canopy management for airflow.",
            "Avoid overhead irrigation.",
            "Remove and destroy infected leaf debris.",
            "Apply preventive copper or mancozeb sprays during humid weather.",
        ],
        "treatment": [
            "Copper-based fungicides or mancozeb applied at 10–14 day intervals.",
            "Systemic fungicides (trifloxystrobin) offer curative activity if applied early.",
        ],
    },

    "Grape___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Prune vines annually and apply wound sealant immediately.",
            "Train shoots to allow maximum light and air penetration.",
            "Scout weekly during the growing season for early disease or pest signs.",
            "Apply dormant copper spray before bud swell.",
        ],
        "treatment": [],
    },

    # ── ORANGE ──────────────────────────────────────────────────────────────
    "Orange___Haunglongbing_(Citrus_greening)": {
        "disease": "Citrus Greening (Huanglongbing / HLB)",
        "pathogen": "Candidatus Liberibacter asiaticus (bacteria), vectored by Asian citrus psyllid",
        "symptoms": (
            "Asymmetric yellow mottling ('blotchy mottle') of leaves; "
            "stunted, upright shoots ('yellow shoot'); "
            "small, lopsided fruit with green areas that don't ripen; "
            "bitter, salty, or off-flavored juice; tree decline over years."
        ),
        "causes": (
            "Bacteria spread by the Asian citrus psyllid (Diaphorina citri). "
            "No cure currently exists; infected trees are a permanent inoculum source."
        ),
        "prevention": [
            "Control psyllid populations with systemic insecticides (imidacloprid) and natural predators.",
            "Inspect nursery stock carefully; buy certified disease-free trees.",
            "Remove and destroy infected trees as soon as diagnosed to prevent spread.",
            "Quarantine measures: do not move plant material from infected areas.",
            "Plant citrus away from existing infected orchards.",
        ],
        "treatment": [
            "No effective cure is currently available commercially.",
            "Thermotherapy (heat treatment at 40–42 °C) shows some experimental promise.",
            "Aggressive psyllid control can slow disease spread.",
            "Remove confirmed HLB-positive trees immediately to protect healthy neighbors.",
        ],
    },

    # ── PEACH ────────────────────────────────────────────────────────────────
    "Peach___Bacterial_spot": {
        "disease": "Peach Bacterial Spot",
        "pathogen": "Xanthomonas arboricola pv. pruni (bacteria)",
        "symptoms": (
            "Small, water-soaked spots on leaves that turn dark and may fall out ('shot hole'); "
            "severe cases cause premature leaf drop; "
            "fruit shows dark, sunken, cracked spots reducing marketability."
        ),
        "causes": (
            "Bacteria overwinter in infected buds and bark; "
            "spread by rain and wind during wet spring weather; "
            "warm (24–29 °C), rainy conditions are most favorable."
        ),
        "prevention": [
            "Plant resistant varieties (e.g., Redhaven, Harbrite).",
            "Apply copper-based bactericides starting at bud swell and through shuck fall.",
            "Prune to improve air circulation and reduce wetness duration.",
            "Avoid overhead irrigation.",
            "Do not plant in frost pockets where late spring frosts cause wounds favorable to infection.",
        ],
        "treatment": [
            "Apply copper hydroxide or copper sulfate sprays preventively and after rain events.",
            "Oxytetracycline (where legally permitted) can reduce bacterial populations.",
            "Remove heavily infected shoots and fruit during the season.",
        ],
    },

    "Peach___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Apply dormant copper spray each winter.",
            "Thin fruit to reduce disease pressure and improve air circulation.",
            "Monitor for peach leaf curl and brown rot during wet springs.",
            "Fertilize according to annual leaf nutrient analysis.",
        ],
        "treatment": [],
    },

    # ── PEPPER (BELL) ────────────────────────────────────────────────────────
    "Pepper,_bell___Bacterial_spot": {
        "disease": "Pepper Bacterial Spot",
        "pathogen": "Xanthomonas euvesicatoria (bacteria)",
        "symptoms": (
            "Water-soaked, greasy-looking leaf spots that turn brown with yellow halos; "
            "defoliation under severe infection; "
            "raised, scabby lesions on fruit; fruit cracks and rots."
        ),
        "causes": (
            "Bacteria spread by rain splash, insects, and contaminated tools. "
            "Favored by warm (24–30 °C), wet, and humid conditions."
        ),
        "prevention": [
            "Use certified disease-free or treated seed.",
            "Plant resistant varieties where available.",
            "Rotate peppers with non-solanaceous crops for 2–3 years.",
            "Avoid overhead irrigation; use drip systems.",
            "Sterilize tools between plants with 10% bleach solution.",
        ],
        "treatment": [
            "Apply copper-based bactericides (copper hydroxide + mancozeb) preventively.",
            "Begin sprays at transplanting and repeat every 5–7 days during wet weather.",
            "Remove and destroy severely infected plants.",
        ],
    },

    "Pepper,_bell___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Maintain soil pH between 6.0 and 7.0.",
            "Apply mulch to reduce soil splash and conserve moisture.",
            "Stake plants to prevent fruit from contacting soil.",
            "Scout weekly for aphids and whiteflies which can vector viruses.",
        ],
        "treatment": [],
    },

    # ── POTATO ───────────────────────────────────────────────────────────────
    "Potato___Early_blight": {
        "disease": "Potato Early Blight",
        "pathogen": "Alternaria solani (fungus)",
        "symptoms": (
            "Dark brown to black, circular lesions with concentric rings ('target board' or 'bull's eye' pattern); "
            "yellow halo surrounds lesions; lower/older leaves affected first; "
            "lesions on tubers appear as dark, sunken, corky patches."
        ),
        "causes": (
            "Favored by warm (24–29 °C), dry weather alternating with humid periods; "
            "plants stressed by drought, nutrient deficiency, or overcropping are most susceptible."
        ),
        "prevention": [
            "Use certified disease-free seed potatoes.",
            "Rotate crops — avoid planting potato or tomato in the same field for 2–3 years.",
            "Maintain adequate soil fertility, especially nitrogen and potassium.",
            "Apply preventive fungicide sprays beginning before disease onset.",
            "Avoid drought stress with consistent irrigation.",
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb, azoxystrobin) at 7–10 day intervals.",
            "Begin applications when plants reach 15 cm height or at first symptom.",
            "Remove and destroy infected lower leaves.",
        ],
    },

    "Potato___Late_blight": {
        "disease": "Potato Late Blight",
        "pathogen": "Phytophthora infestans (water mold / oomycete)",
        "symptoms": (
            "Irregular, water-soaked, pale green to brown lesions on leaves; "
            "white, downy sporulation on leaf undersides in humid conditions; "
            "entire fields can collapse within days in favorable weather; "
            "tuber rot with reddish-brown granular internal discoloration."
        ),
        "causes": (
            "Thrives in cool (10–21 °C), wet, and foggy conditions. "
            "Spreads rapidly via wind-blown sporangia. "
            "Historically caused the Irish Potato Famine (1845–49)."
        ),
        "prevention": [
            "Plant certified disease-free seed potatoes.",
            "Use late blight-resistant varieties (e.g., Defender, Elba).",
            "Avoid planting near cull piles or volunteer potato plants.",
            "Hill up soil around stems to protect tubers.",
            "Apply preventive fungicide program starting before disease onset.",
        ],
        "treatment": [
            "Apply fungicides (mancozeb, chlorothalonil, cymoxanil, metalaxyl) on a 5–7 day schedule.",
            "Metalaxyl/mefenoxam products are highly effective but resistance can develop — rotate modes of action.",
            "Destroy (burn or deep bury) infected haulm at harvest.",
            "Do not store infected tubers; they will rot and spread disease.",
        ],
    },

    "Potato___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Always use certified, disease-free seed potatoes.",
            "Hill rows properly to protect tubers from sunlight and disease.",
            "Monitor weekly for early blight and late blight under humid conditions.",
            "Maintain balanced irrigation to avoid drought stress.",
        ],
        "treatment": [],
    },

    # ── RASPBERRY ────────────────────────────────────────────────────────────
    "Raspberry___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Remove and destroy old canes after fruiting (floricanes) to reduce disease inoculum.",
            "Maintain well-drained, slightly acidic soil (pH 5.5–6.5).",
            "Ensure adequate plant spacing for air circulation.",
            "Apply dormant oil spray to control spider mites and scale insects.",
        ],
        "treatment": [],
    },

    # ── SOYBEAN ──────────────────────────────────────────────────────────────
    "Soybean___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Rotate with non-legume crops (corn, wheat) to reduce cyst nematode and white mold pressure.",
            "Use SCN-resistant varieties in fields with known soybean cyst nematode.",
            "Scout fields regularly for sudden death syndrome, frogeye leaf spot, and aphids.",
            "Inoculate with Bradyrhizobium japonicum for optimal nitrogen fixation.",
        ],
        "treatment": [],
    },

    # ── SQUASH ───────────────────────────────────────────────────────────────
    "Squash___Powdery_mildew": {
        "disease": "Squash Powdery Mildew",
        "pathogen": "Podosphaera xanthii / Erysiphe cichoracearum (fungi)",
        "symptoms": (
            "White, powdery fungal patches on upper leaf surfaces; "
            "affected leaves yellow and die; "
            "severe infection stunts plant and reduces yield."
        ),
        "causes": (
            "Favored by warm (20–27 °C), dry days with high relative humidity at night. "
            "Does NOT require free moisture for spore germination — unlike most fungal diseases."
        ),
        "prevention": [
            "Plant resistant or tolerant squash varieties.",
            "Ensure adequate spacing between plants for airflow.",
            "Avoid excessive nitrogen fertilization which promotes susceptible new growth.",
            "Remove and destroy infected plant material promptly.",
        ],
        "treatment": [
            "Apply sulfur-based fungicides or potassium bicarbonate sprays.",
            "Use systemic fungicides (myclobutanil, trifloxystrobin) for rapid suppression.",
            "Neem oil or horticultural oil sprays can also reduce spore germination.",
            "Repeat applications every 7–10 days as needed.",
        ],
    },

    # ── STRAWBERRY ───────────────────────────────────────────────────────────
    "Strawberry___Leaf_scorch": {
        "disease": "Strawberry Leaf Scorch",
        "pathogen": "Diplocarpon earlianum (fungus)",
        "symptoms": (
            "Small, irregular, dark purple spots on upper leaf surface; "
            "spots may coalesce giving leaves a 'scorched' appearance; "
            "severe infection causes defoliation and reduced plant vigor."
        ),
        "causes": (
            "Fungal spores overwinter in infected leaves. "
            "Spread by rain splash; favored by warm, wet conditions."
        ),
        "prevention": [
            "Remove and destroy old infected leaves in autumn.",
            "Choose resistant varieties (e.g., Allstar, Earliglow).",
            "Use drip irrigation instead of overhead sprinklers.",
            "Renovate matted-row plantings annually to reduce inoculum buildup.",
        ],
        "treatment": [
            "Apply fungicides (captan, myclobutanil) at 2-week intervals from early spring.",
            "Remove heavily infected leaves by hand.",
            "Apply fungicide after renovation mowing.",
        ],
    },

    "Strawberry___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Renovate beds annually by mowing, thinning, and applying fertilizer after harvest.",
            "Remove runners to keep plant spacing open and airy.",
            "Apply straw mulch in winter to protect crowns and reduce splash.",
            "Monitor for gray mold (Botrytis) during bloom and wet weather.",
        ],
        "treatment": [],
    },

    # ── TOMATO ───────────────────────────────────────────────────────────────
    "Tomato___Bacterial_spot": {
        "disease": "Tomato Bacterial Spot",
        "pathogen": "Xanthomonas euvesicatoria (bacteria)",
        "symptoms": (
            "Small, water-soaked spots on leaves that enlarge, turn brown, and develop yellow halos; "
            "shot-hole appearance as infected tissue falls out; "
            "raised, scabby, brown lesions on fruit reducing marketability."
        ),
        "causes": (
            "Bacteria spread by rain, irrigation, and contaminated tools or transplants. "
            "Warm (24–30 °C), wet, and windy conditions favor rapid spread."
        ),
        "prevention": [
            "Use certified pathogen-free seed or hot-water treat seed at 52 °C for 30 minutes.",
            "Avoid overhead irrigation; use drip systems.",
            "Rotate with non-solanaceous crops for 2–3 years.",
            "Sterilize tools and equipment with 10% bleach or 70% ethanol.",
            "Plant in well-drained soil with good air circulation.",
        ],
        "treatment": [
            "Apply copper-based bactericides (copper hydroxide + mancozeb) at 5–7 day intervals during wet periods.",
            "Remove and destroy severely infected plant material.",
            "Copper resistance is common — rotate to plant activators (Actigard) if resistance is suspected.",
        ],
    },

    "Tomato___Early_blight": {
        "disease": "Tomato Early Blight",
        "pathogen": "Alternaria solani (fungus)",
        "symptoms": (
            "Dark brown concentric-ring ('target board') lesions on lower, older leaves first; "
            "yellow halo surrounds lesions; progresses upward; "
            "stem lesions (collar rot) may kill seedlings; "
            "sunken, dark, leathery lesions at stem end of fruit."
        ),
        "causes": (
            "Survives in infected debris and soil. "
            "Warm (24–29 °C) weather with alternating wet and dry periods favors infection. "
            "Stressed or senescent plants are most susceptible."
        ),
        "prevention": [
            "Use certified disease-free transplants.",
            "Rotate tomatoes with non-solanaceous crops for 2–3 years.",
            "Mulch heavily to prevent soil splash onto lower leaves.",
            "Remove lower leaves as plants grow to improve airflow.",
            "Avoid drought stress — use consistent drip irrigation.",
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb, azoxystrobin, or copper) at 7–10 day intervals.",
            "Remove and destroy infected leaves.",
            "Begin applications preventively once plants are established.",
        ],
    },

    "Tomato___Late_blight": {
        "disease": "Tomato Late Blight",
        "pathogen": "Phytophthora infestans (oomycete)",
        "symptoms": (
            "Large, irregular, water-soaked, pale green to dark brown lesions on leaves; "
            "white downy sporulation on leaf undersides in humid conditions; "
            "brown, firm, greasy lesions on stems and fruit; "
            "plant collapse can occur within days under favorable conditions."
        ),
        "causes": (
            "Cool (10–21 °C), wet weather with >90% humidity. "
            "Spreads rapidly via wind-borne sporangia over long distances."
        ),
        "prevention": [
            "Plant resistant varieties (Mountain Merit, Iron Lady, Defiant).",
            "Avoid planting near potatoes — both are hosts.",
            "Space plants widely for air circulation; stake and prune for canopy openness.",
            "Apply preventive fungicide program in cool, wet weather forecasts.",
            "Remove volunteer potato and tomato plants from the area.",
        ],
        "treatment": [
            "Apply fungicides (mancozeb, chlorothalonil, cymoxanil, or phosphorous acid) at 5–7 day intervals.",
            "Once established, late blight is very difficult to control — prevention is critical.",
            "Remove and bag (do not compost) all infected plant material immediately.",
            "Destroy entire plant if the stem is infected.",
        ],
    },

    "Tomato___Leaf_Mold": {
        "disease": "Tomato Leaf Mold",
        "pathogen": "Passalora fulva (syn. Fulvia fulva) (fungus)",
        "symptoms": (
            "Pale yellow spots on upper leaf surface; "
            "corresponding olive-green to gray velvety fungal growth on leaf undersides; "
            "infected leaves curl, wither, and drop; primarily affects greenhouse tomatoes."
        ),
        "causes": (
            "High humidity (>85%) and moderate temperatures (20–25 °C) are critical. "
            "Spreads by airborne conidia; common in greenhouses and high tunnels."
        ),
        "prevention": [
            "Reduce greenhouse humidity by ventilation, heating, and spacing plants adequately.",
            "Use drip irrigation and avoid wetting foliage.",
            "Plant resistant varieties (e.g., Jasper, Plum Regal).",
            "Remove infected leaves promptly.",
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb, copper, or systemic triazoles) at first sign.",
            "Improve ventilation immediately — humidity control is the most effective intervention.",
            "Remove and destroy infected leaves.",
        ],
    },

    "Tomato___Septoria_leaf_spot": {
        "disease": "Septoria Leaf Spot",
        "pathogen": "Septoria lycopersici (fungus)",
        "symptoms": (
            "Numerous small (2–4 mm), circular spots with dark brown margins and lighter tan/gray centers; "
            "tiny black fruiting bodies (pycnidia) visible in lesion centers; "
            "lower leaves affected first, progresses rapidly upward; severe defoliation."
        ),
        "causes": (
            "Favored by warm (20–25 °C), wet, and humid conditions. "
            "Spreads by rain splash and infected crop debris."
        ),
        "prevention": [
            "Rotate tomatoes away from solanaceous crops for 2–3 years.",
            "Use mulch to prevent rain splash from soil onto lower leaves.",
            "Remove lower leaves to improve airflow.",
            "Stake plants to keep foliage off the ground.",
        ],
        "treatment": [
            "Apply fungicides (chlorothalonil, mancozeb, copper, azoxystrobin) at 7–10 day intervals.",
            "Remove infected lower leaves with gloves and dispose in sealed bags.",
            "Begin treatment early — once most lower leaves are affected, control is difficult.",
        ],
    },

    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease": "Two-Spotted Spider Mite Infestation",
        "pathogen": "Tetranychus urticae (arachnid pest — not a fungal/bacterial disease)",
        "symptoms": (
            "Fine stippling (tiny yellow or white specks) on upper leaf surfaces; "
            "bronze or gray discoloration of leaves; "
            "fine silk webbing on leaf undersides; "
            "severe infestation causes leaf desiccation and drop."
        ),
        "causes": (
            "Hot (>27 °C), dry, and dusty conditions favor rapid mite population explosions. "
            "Broad-spectrum insecticide use destroys natural predators, triggering mite outbreaks."
        ),
        "prevention": [
            "Maintain adequate irrigation — water-stressed plants are far more susceptible.",
            "Avoid using broad-spectrum insecticides that kill beneficial predatory mites.",
            "Introduce or conserve natural predators (Phytoseiid predatory mites, lacewings).",
            "Dust control on field margins reduces mite dispersal.",
        ],
        "treatment": [
            "Apply miticides/acaricides (abamectin, bifenazate, spiromesifen) targeting leaf undersides.",
            "Rotate miticide classes to prevent resistance.",
            "Insecticidal soap or horticultural oil sprays can reduce populations if detected early.",
            "Release biological control agents (Phytoseiulus persimilis) in greenhouse settings.",
        ],
    },

    "Tomato___Target_Spot": {
        "disease": "Tomato Target Spot",
        "pathogen": "Corynespora cassiicola (fungus)",
        "symptoms": (
            "Brown, circular lesions with concentric rings (target pattern); "
            "yellow halo may surround lesions; "
            "lesions on stems, petioles, and fruit; "
            "causes defoliation and reduced yield."
        ),
        "causes": (
            "Warm (24–30 °C), humid conditions. "
            "Survives in soil and plant debris; spreads by wind and water splash."
        ),
        "prevention": [
            "Use resistant varieties where available.",
            "Avoid overhead irrigation; mulch to prevent soil splash.",
            "Remove lower leaves and maintain good airflow.",
            "Rotate with non-solanaceous crops.",
        ],
        "treatment": [
            "Apply fungicides (azoxystrobin, chlorothalonil, or mancozeb) on a 7–10 day schedule.",
            "Systemic fungicides (flutriafol, trifloxystrobin) show good efficacy.",
            "Remove and destroy severely infected plant material.",
        ],
    },

    "Tomato___Tomato_mosaic_virus": {
        "disease": "Tomato Mosaic Virus (ToMV)",
        "pathogen": "Tomato mosaic virus (Tobamovirus genus)",
        "symptoms": (
            "Mosaic pattern of light and dark green (or yellow) on leaves; "
            "leaf distortion, curling, and blistering; "
            "stunted plant growth; fruit may show internal browning ('brown wall')."
        ),
        "causes": (
            "Highly stable virus — can survive on surfaces, tools, and clothing for extended periods. "
            "Spreads mechanically (hands, tools, grafting) and through infected seed. "
            "NOT transmitted by insects (unlike TYLCV)."
        ),
        "prevention": [
            "Use ToMV-resistant varieties (carrying Tm-2² resistance gene).",
            "Start with certified virus-free seed or hot-water treat seed.",
            "Wash hands thoroughly before working with plants; avoid smoking near tomatoes.",
            "Sterilize all tools with 10% bleach before and between use.",
            "Remove infected plants immediately to prevent mechanical spread.",
        ],
        "treatment": [
            "No chemical cure exists for viral diseases.",
            "Remove and destroy infected plants immediately.",
            "Strict hygiene (tool sterilization, handwashing) prevents further spread.",
        ],
    },

    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "pathogen": "Tomato yellow leaf curl virus (Begomovirus genus), vectored by Bemisia tabaci (whitefly)",
        "symptoms": (
            "Upward curling and cupping of young leaves; "
            "yellowing (chlorosis) of leaf margins and young leaves; "
            "stunted, bushy appearance ('witches'-broom'); "
            "severe flower and fruit drop; infected plants rarely produce marketable yield."
        ),
        "causes": (
            "Virus is transmitted persistently by the silverleaf whitefly (Bemisia tabaci). "
            "Cannot spread by contact or tools — whitefly control is the primary management lever."
        ),
        "prevention": [
            "Plant TYLCV-resistant varieties (e.g., Shanty, Hazera series).",
            "Use reflective silver mulches to repel whiteflies.",
            "Install 50-mesh insect screens on greenhouse openings.",
            "Apply systemic insecticides (imidacloprid as drench) at transplanting.",
            "Remove weeds that serve as alternative whitefly hosts.",
            "Establish a plant-free period between crops to break the whitefly/virus cycle.",
        ],
        "treatment": [
            "No chemical cure for the virus itself.",
            "Aggressively control whitefly populations with insecticides (spirotetramat, pyriproxyfen, insecticidal soap).",
            "Remove and destroy symptomatic plants early before whiteflies abandon them and spread to healthy plants.",
            "Rotate insecticide classes to prevent whitefly resistance.",
        ],
    },

    "Tomato___healthy": {
        "disease": "Healthy",
        "pathogen": None,
        "symptoms": "No disease symptoms detected. Plant appears healthy.",
        "causes": "N/A",
        "prevention": [
            "Rotate with non-solanaceous crops every 2–3 years.",
            "Use mulch to reduce soil splash and conserve moisture.",
            "Prune suckers and lower leaves regularly for airflow.",
            "Scout weekly for early signs of blight, viral symptoms, or mite infestations.",
        ],
        "treatment": [],
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_tips(class_name: str) -> Optional[Dict[str, Any]]:
    """Return the prevention/treatment dictionary for a given class name.

    Args:
        class_name: The ImageFolder class name (e.g., 'Tomato___Late_blight').

    Returns:
        A dict with keys: disease, pathogen, symptoms, causes, prevention (list), treatment (list).
        Returns None if the class name is not found.
    """
    return PREVENTION_TIPS.get(class_name, None)


def format_tips(class_name: str) -> str:
    """Return a human-readable string of all tips for a given class.

    Args:
        class_name: The ImageFolder class name.

    Returns:
        A formatted multi-line string, or a 'not found' message.
    """
    tips = get_tips(class_name)
    if tips is None:
        return f"No information found for class: '{class_name}'"

    lines = [
        f"Disease       : {tips['disease']}",
        f"Pathogen      : {tips['pathogen'] or 'N/A'}",
        "",
        f"Symptoms      : {tips['symptoms']}",
        "",
        f"Causes        : {tips['causes']}",
        "",
        "Prevention:",
    ]
    for i, tip in enumerate(tips["prevention"], 1):
        lines.append(f"  {i}. {tip}")

    if tips["treatment"]:
        lines.append("")
        lines.append("Treatment:")
        for i, tip in enumerate(tips["treatment"], 1):
            lines.append(f"  {i}. {tip}")
    else:
        lines.append("")
        lines.append("Treatment: Plant is healthy — no treatment required.")

    return "\n".join(lines)


def get_all_class_names() -> list:
    """Return a sorted list of all supported class names."""
    return sorted(PREVENTION_TIPS.keys())
