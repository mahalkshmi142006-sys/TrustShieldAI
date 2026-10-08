const $ = x => document.getElementById(x);

let S = JSON.parse(
localStorage.tsa ||
'{"t":0,"l":0,"m":0,"a":[]}'
);

/* ============================================================
STORAGE
============================================================ */

function save() {
localStorage.tsa = JSON.stringify(S);
refresh();
}

/* ============================================================
DASHBOARD REFRESH
============================================================ */

function refresh() {

```
$("total").textContent = S.t;
$("lc").textContent = S.l;
$("mc").textContent = S.m;

let h = S.a.length
    ? S.a.map(x =>
        `<div class="claim">
            <b>${x[0]}</b><br>
            <small>${x[1]}</small>
        </div>`
    ).join("")
    : "No checks yet.";

$("recent").innerHTML = h;
$("acts").innerHTML = h;

fetch("/api/health")
    .then(r => r.json())
    .then(d => {

        $("ec").textContent =
            d.evidence_count;

        $("ev").textContent =
            d.evidence_count +
            " evidence records";

    })
    .catch(() => {

        $("ev").textContent =
            "API unavailable";

    });
```

}

/* ============================================================
TABS
============================================================ */

function tab(id) {

```
document
    .querySelectorAll(".tab")
    .forEach(x =>
        x.classList.remove("active")
    );

$(id).classList.add("active");

document
    .querySelectorAll("nav button")
    .forEach(x =>
        x.classList.toggle(
            "active",
            x.dataset.t === id
        )
    );
```

}

document
.querySelectorAll("nav button")
.forEach(
x =>
x.onclick =
() => tab(x.dataset.t)
);

/* ============================================================
ACTIVITY
============================================================ */

function add(a, b) {

```
S.t++;

S.a.unshift([
    a,
    b
]);

S.a =
    S.a.slice(0, 20);

save();
```

}

/* ============================================================
LLM HALLUCINATION ANALYSIS
============================================================ */

async function llm() {

```
let q =
    $("q").value.trim();

let r =
    $("r").value.trim();

if (!q || !r)
    return alert(
        "Enter both question and response."
    );

$("lr").textContent =
    "Analyzing…";

try {

    let x =
        await fetch(
            "/api/analyze",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        question: q,
                        response: r
                    })
            }
        );

    let d =
        await x.json();

    if (!x.ok)
        throw Error(d.error);

    $("lr").innerHTML = `

        <b class="${
            d.hallucination_risk < 30
                ? "good"
                : d.hallucination_risk < 60
                    ? "warn"
                    : "bad"
        }">
            ${d.assessment}
        </b>

        <div>

            <span class="metric">
                Trust
                <b>${d.trust_score}%</b>
            </span>

            <span class="metric">
                Risk
                <b>${d.hallucination_risk}%</b>
            </span>

            <span class="metric">
                Confidence
                <b>${d.confidence}%</b>
            </span>

        </div>

        <p>
            Q–A consistency:
            <b class="${
                d.question_answer_consistent
                    ? "good"
                    : "bad"
            }">
                ${
                    d.question_answer_consistent
                        ? "CONSISTENT"
                        : "MISMATCH"
                }
            </b>
        </p>
    `;

    $("claims").innerHTML =
        d.claims.map(c =>
            `<div class="claim">
                <b>${c.verdict}</b>
                — ${c.claim}<br>
                <small>
                    ${c.evidence_id}
                    · similarity ${c.similarity}
                    · ${c.evidence}
                </small>
            </div>`
        ).join("");

    S.l++;

    add(
        "LLM verification",
        d.assessment
    );

} catch (e) {

    $("lr").textContent =
        "Error: " + e.message;
}
```

}

/* ============================================================
DEEPFAKE / MEDIA ANALYSIS
STAGE 3.6 EXPLAINABLE VERIFICATION
============================================================ */

async function media() {

```
let f =
    $("mf").files[0];

if (!f)
    return alert(
        "Choose a file."
    );

let fd =
    new FormData();

fd.append(
    "file",
    f
);

$("mr").innerHTML =
    "Analyzing media…";

try {

    let x =
        await fetch(
            "/api/analyze-media",
            {
                method: "POST",
                body: fd
            }
        );

    let d =
        await x.json();

    if (!x.ok)
        throw Error(d.error);


    /* ----------------------------------------------------
       PRIMARY RESULT
    ---------------------------------------------------- */

    let risk =
        Number(
            d.stage35_deepfake_risk ??
            d.deepfake_risk ??
            0
        );

    let authenticity =
        Number(
            d.stage35_authenticity_score ??
            d.authenticity_score ??
            (100 - risk)
        );


    let verdict =
        d.stage35_verdict ||
        (
            risk < 40
                ? "LIKELY AUTHENTIC"
                : risk < 70
                    ? "REQUIRES REVIEW"
                    : "LIKELY DEEPFAKE"
        );


    let verdictClass =
        risk < 40
            ? "good"
            : risk < 70
                ? "warn"
                : "bad";


    /* ----------------------------------------------------
       MODEL PERFORMANCE
    ---------------------------------------------------- */

    let accuracy =
        d.stage35_test_accuracy != null
            ? (
                Number(
                    d.stage35_test_accuracy
                ) * 100
            ).toFixed(2)
            : "—";


    let precision =
        d.stage35_test_precision != null
            ? (
                Number(
                    d.stage35_test_precision
                ) * 100
            ).toFixed(2)
            : "—";


    let recall =
        d.stage35_test_recall != null
            ? (
                Number(
                    d.stage35_test_recall
                ) * 100
            ).toFixed(2)
            : "—";


    let f1 =
        d.stage35_test_f1 != null
            ? (
                Number(
                    d.stage35_test_f1
                ) * 100
            ).toFixed(2)
            : "—";


    let auc =
        d.stage35_test_roc_auc != null
            ? (
                Number(
                    d.stage35_test_roc_auc
                ) * 100
            ).toFixed(2)
            : "—";


    /* ----------------------------------------------------
       EXPLANATION
    ---------------------------------------------------- */

    let explanations =
        Array.isArray(
            d.stage35_explanation
        )
            ? d.stage35_explanation
            : [];


    let explanationHTML =
        explanations.length
            ? explanations.map(
                item =>
                    `<div class="claim">
                        ✓ ${item}
                    </div>`
            ).join("")
            : `
                <div class="claim">
                    ✓ Forensic analysis completed.
                </div>
            `;


    /* ----------------------------------------------------
       TOP FORENSIC FEATURES
    ---------------------------------------------------- */

    let features =
        Array.isArray(
            d.stage35_top_features
        )
            ? d.stage35_top_features
            : [];


    let featureHTML =
        features.length
            ? features.slice(0, 5).map(
                item => {

                    let importance =
                        Number(
                            item.importance
                        ) * 100;

                    return `
                        <div class="claim">

                            <b>
                                ${item.feature}
                            </b>

                            <br>

                            <small>
                                Importance:
                                ${importance.toFixed(2)}%
                            </small>

                        </div>
                    `;
                }
            ).join("")
            : `
                <div class="claim">
                    Feature importance unavailable.
                </div>
            `;


    /* ----------------------------------------------------
       FINAL DASHBOARD RESULT
    ---------------------------------------------------- */

    $("mr").innerHTML = `

        <div>

            <b class="${verdictClass}">
                ${verdict}
            </b>

        </div>


        <div>

            <span class="metric">
                Deepfake Risk
                <b>${risk}%</b>
            </span>

            <span class="metric">
                Authenticity
                <b>${authenticity}%</b>
            </span>

            <span class="metric">
                Faces
                <b>
                    ${
                        d.face_count ??
                        "—"
                    }
                </b>
            </span>

        </div>


        <hr>


        <h3>
            Explainable Verification
        </h3>

        ${explanationHTML}


        <h3>
            Stage 3.5 Model Performance
        </h3>

        <div>

            <span class="metric">
                Accuracy
                <b>${accuracy}%</b>
            </span>

            <span class="metric">
                Precision
                <b>${precision}%</b>
            </span>

            <span class="metric">
                Recall
                <b>${recall}%</b>
            </span>

            <span class="metric">
                F1
                <b>${f1}%</b>
            </span>

            <span class="metric">
                ROC-AUC
                <b>${auc}%</b>
            </span>

        </div>


        <h3>
            Top Forensic Features
        </h3>

        ${featureHTML}


        <h3>
            Model
        </h3>

        <p>
            <b>
                ${
                    d.stage35_model ||
                    "deepfake_rf_stage3_5.joblib"
                }
            </b>
        </p>

        <p>
            ${
                d.stage35_note ||
                d.method ||
                "Explainable forensic verification completed."
            }
        </p>

    `;


    /* ----------------------------------------------------
       ACTIVITY LOG
    ---------------------------------------------------- */

    S.m++;

    add(
        "Deepfake verification",
        `${verdict} — ${risk}% risk`
    );


} catch (e) {

    $("mr").innerHTML =
        `<b class="bad">
            Error: ${e.message}
        </b>`;
}
```

}

/* ============================================================
IDENTITY VERIFICATION
============================================================ */

async function identity() {

```
let a =
    $("ref").files[0];

let b =
    $("pr").files[0];

if (!a || !b)
    return alert(
        "Choose both images."
    );

let fd =
    new FormData();

fd.append(
    "reference",
    a
);

fd.append(
    "probe",
    b
);

$("ir").textContent =
    "Comparing…";

try {

    let x =
        await fetch(
            "/api/verify-identity",
            {
                method: "POST",
                body: fd
            }
        );

    let d =
        await x.json();

    if (!x.ok)
        throw Error(d.error);

    $("ir").innerHTML = `

        <b class="${
            d.status === "MATCH"
                ? "good"
                : d.status === "NO MATCH"
                    ? "bad"
                    : "warn"
        }">
            ${d.status}
        </b>

        <div class="metric">
            Similarity
            <b>
                ${d.similarity}%
            </b>
        </div>

        <p>
            ${d.method || d.message}
        </p>
    `;

    add(
        "Identity verification",
        d.status
    );

} catch (e) {

    $("ir").textContent =
        "Error: " + e.message;
}
```

}

/* ============================================================
INITIAL LOAD
============================================================ */

refresh();
