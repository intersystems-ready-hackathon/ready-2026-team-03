/// Pre-op screening guideline for a clinical specialty (e.g. Endocrine
/// Surgery). One row per specialty; the agent uses the `Content` text to
/// check medications / allergies / risk factors against the procedure type
/// and to produce generic pre-op advice.
Class Data.SpecialtyGuide Extends %Persistent
{

/// Specialty identifier such as "ST-ENDO". Natural primary key.
Property SpecialtyID As %String(MAXLEN = 32) [ Required ];

/// Human-readable specialty name (e.g. "Endocrine Surgery").
Property SpecialtyName As %String(MAXLEN = 100);

/// Comma-separated list of procedures covered by this specialty. Lets the
/// agent map a procedure name back to its specialty without parsing the
/// full guide text.
Property Procedures As %String(MAXLEN = 1000);

/// Full guide text — risk factors, medications, allergies, and prompts.
/// Held inline as a long string so it can be returned in one SQL SELECT.
Property Content As %String(MAXLEN = 10000);

Property CreatedAt As %TimeStamp [ InitialExpression = {$ZDATETIME($HOROLOG, 3)} ];

Property UpdatedAt As %TimeStamp;

/// SpecialtyID is the natural unique key.
Index SpecialtyIDIndex On SpecialtyID [ PrimaryKey, Unique ];

/// Seed every known specialty guide. Idempotent: re-running silently
/// no-ops on rows whose `SpecialtyID` is already present (the unique
/// index makes %Save() fail without raising).
ClassMethod Populate() As %Status
{
    Do ..SaveGuide(
        "ST-CARD", "Cardiothoracic Surgery",
        "Coronary artery bypass graft, Aortic valve replacement, Mitral valve repair, Lobectomy via thoracotomy, Video-assisted thoracoscopic wedge resection",
        ..CardContent())
    Do ..SaveGuide(
        "ST-ENDO", "Endocrine Surgery",
        "Total thyroidectomy, Parathyroidectomy, Adrenalectomy, Hemithyroidectomy, Thyroid lobectomy",
        ..EndoContent())
    Do ..SaveGuide(
        "ST-GENS", "General Surgery",
        "Laparoscopic cholecystectomy, Open inguinal hernia repair, Laparoscopic appendicectomy, Colectomy, Haemorrhoidectomy",
        ..GensContent())
    Do ..SaveGuide(
        "ST-ORTH", "Orthopaedics",
        "Total hip replacement, Total knee replacement, ACL reconstruction, Rotator cuff repair, Ankle fracture fixation",
        ..OrthContent())
    Do ..SaveGuide(
        "ST-UROL", "Urology",
        "Transurethral resection of prostate, Ureteroscopy and laser lithotripsy, Circumcision, Nephrectomy, Cystoscopy with bladder biopsy",
        ..UrolContent())
    Return $$$OK
}

ClassMethod SaveGuide(SpecialtyID As %String, SpecialtyName As %String, Procedures As %String, Content As %String) As %Status
{
    Set guide = ##class(Data.SpecialtyGuide).%New()
    Set guide.SpecialtyID = SpecialtyID
    Set guide.SpecialtyName = SpecialtyName
    Set guide.Procedures = Procedures
    Set guide.Content = Content
    Do guide.%Save()
    Return $$$OK
}

/// Cardiothoracic Surgery — inlined from specialty_guides_txt/ST-CARD_cardiothoracic_surgery.txt.
ClassMethod CardContent() As %String
{
    Quit $listtostring($listbuild(
        "Specialty ID: ST-CARD",
        "Specialty: Cardiothoracic Surgery",
        "",
        "Purpose",
        "Synthetic chatbot screening guide for patients scheduled for cardiothoracic procedures.",
        "",
        "Relevant procedures in dataset",
        "- Coronary artery bypass graft",
        "- Aortic valve replacement",
        "- Mitral valve repair",
        "- Lobectomy via thoracotomy",
        "- Video-assisted thoracoscopic wedge resection",
        "",
        "Risk factors to ask about",
        "- Angina, previous heart attack, heart failure, cardiomyopathy, heart murmur, rheumatic fever",
        "- Pacemaker, implantable defibrillator, irregular heartbeat, heart block",
        "- Uncontrolled hypertension or current antihypertensive therapy",
        "- Stroke or transient ischaemic attack",
        "- Blood clots, DVT, PE, recent clots within 3 months",
        "- Significant blood disorders, excessive bleeding or bruising",
        "- Shortness of breath, COPD, asthma, obstructive sleep apnoea, recent chest infection",
        "- Reduced exercise tolerance and ability to climb stairs",
        "- Previous ICU admission or significant anaesthetic complications",
        "- Smoking status, frailty, mobility limitations, and need for hospital transport",
        "",
        "Notable medications to ask about",
        "- Anticoagulants: warfarin, apixaban, rivaroxaban, edoxaban, dabigatran, heparin",
        "- Antiplatelets: aspirin, clopidogrel, ticagrelor, prasugrel",
        "- Beta blockers, ACE inhibitors, ARBs, calcium-channel blockers, nitrates, diuretics",
        "- Diabetes medicines including insulin and SGLT2 inhibitors",
        "- Steroids or immunosuppressants",
        "- More than 3 prescribed medicines",
        "",
        "Allergies / reactions to ask about",
        "- Medicine allergies",
        "- Latex, plasters/adhesives, iodine/chlorhexidine, food allergies",
        "- Previous anaphylaxis",
        "",
        "Other relevant chatbot prompts",
        "- Confirm whether the patient has a cardiac device and ask for device card/details.",
        "- Ask about chest pain at rest, severe chest pain, worsening breathlessness, recent blackouts, or recent infection.",
        "- Ask about recent admissions, recent procedures, and whether the patient has had a general anaesthetic in the past year."
    ), $char(10))
}

/// Endocrine Surgery — inlined from specialty_guides_txt/ST-ENDO_endocrine_surgery.txt.
ClassMethod EndoContent() As %String
{
    Quit $listtostring($listbuild(
        "Specialty ID: ST-ENDO",
        "Specialty: Endocrine Surgery",
        "",
        "Purpose",
        "Synthetic chatbot screening guide for patients scheduled for endocrine surgical procedures.",
        "",
        "Relevant procedures in dataset",
        "- Total thyroidectomy",
        "- Parathyroidectomy",
        "- Adrenalectomy",
        "- Hemithyroidectomy",
        "- Thyroid lobectomy",
        "",
        "Risk factors to ask about",
        "- Thyroid disease: overactive thyroid, underactive thyroid, thyroid storm history, compressive symptoms",
        "- Diabetes and current treatment",
        "- Kidney disease, especially for parathyroid disease or calcium disturbance",
        "- Hypertension, palpitations, arrhythmias, blackouts, severe dizziness",
        "- Steroid dependence or adrenal insufficiency",
        "- Voice changes, swallowing difficulty, breathing difficulty, neck swelling",
        "- Restricted neck movement, limited mouth opening, previous tracheotomy",
        "- Obstructive sleep apnoea, asthma, COPD, shortness of breath",
        "- Previous anaesthetic complications or difficult airway history",
        "",
        "Notable medications to ask about",
        "- Levothyroxine, carbimazole, propylthiouracil, liothyronine",
        "- Beta blockers used for thyroid symptoms",
        "- Calcium supplements, vitamin D, cinacalcet",
        "- Steroids, hydrocortisone, prednisolone, immunosuppressants",
        "- Anticoagulants/antiplatelets",
        "- Diabetes medicines including insulin and oral agents",
        "",
        "Allergies / reactions to ask about",
        "- Medicines, latex, plasters/adhesives, antiseptics, food allergies",
        "- Previous anaphylaxis",
        "",
        "Other relevant chatbot prompts",
        "- For thyroid procedures, ask about voice hoarseness, swallowing, breathing, and ability to lie flat.",
        "- For parathyroid procedures, ask about kidney stones, fractures, bone pain, confusion, or calcium-related symptoms.",
        "- For adrenal procedures, ask about episodic headaches, sweating, palpitations, severe hypertension, or steroid replacement."
    ), $char(10))
}

/// General Surgery — inlined from specialty_guides_txt/ST-GENS_general_surgery.txt.
ClassMethod GensContent() As %String
{
    Quit $listtostring($listbuild(
        "Specialty ID: ST-GENS",
        "Specialty: General Surgery",
        "",
        "Purpose",
        "Synthetic chatbot screening guide for patients scheduled for general surgical procedures.",
        "",
        "Relevant procedures in dataset",
        "- Laparoscopic cholecystectomy",
        "- Open inguinal hernia repair",
        "- Laparoscopic appendicectomy",
        "- Colectomy",
        "- Haemorrhoidectomy",
        "",
        "Risk factors to ask about",
        "- Heart disease, uncontrolled hypertension, stroke/TIA, DVT/PE, bleeding disorders",
        "- Asthma, COPD, obstructive sleep apnoea, recent respiratory infection",
        "- Diabetes and current glucose control",
        "- Kidney disease, liver disease, jaundice, hepatitis",
        "- Significant reflux disease, hiatus hernia, heartburn",
        "- Previous abdominal surgery, stomas, hernias, bowel disease where relevant",
        "- Reduced functional capacity, frailty, mobility limitations",
        "- Previous anaesthetic complications or family history of anaesthetic problems",
        "- Physical impairment, hearing/visual impairment, interpreter needs",
        "",
        "Notable medications to ask about",
        "- Anticoagulants and antiplatelets: warfarin, DOACs, heparin, aspirin, clopidogrel, ticagrelor",
        "- Diabetes medicines including insulin, oral agents, and SGLT2 inhibitors",
        "- Steroids or immunosuppressants",
        "- Opioids or chronic pain medicines",
        "- More than 3 prescribed medicines",
        "",
        "Allergies / reactions to ask about",
        "- Medicines, latex, plasters/adhesives, antiseptics, food allergies",
        "- Previous anaphylaxis",
        "",
        "Other relevant chatbot prompts",
        "- Ask about abdominal symptoms relevant to procedure: vomiting, jaundice, bowel habit changes, bleeding, infection symptoms.",
        "- Ask whether the patient can tolerate lying flat and whether they have reflux/aspiration risks.",
        "- Ask about previous surgery/general anaesthetic in the past year."
    ), $char(10))
}

/// Orthopaedics — inlined from specialty_guides_txt/ST-ORTH_orthopaedics.txt.
ClassMethod OrthContent() As %String
{
    Quit $listtostring($listbuild(
        "Specialty ID: ST-ORTH",
        "Specialty: Orthopaedics",
        "",
        "Purpose",
        "Synthetic chatbot screening guide for patients scheduled for orthopaedic procedures.",
        "",
        "Relevant procedures in dataset",
        "- Total hip replacement",
        "- Total knee replacement",
        "- ACL reconstruction",
        "- Rotator cuff repair",
        "- Ankle fracture fixation",
        "",
        "Risk factors to ask about",
        "- Mobility limitation, walking aids, wheelchair use, falls risk, home support needs",
        "- DVT/PE history and recent clots within 3 months",
        "- Heart disease, hypertension, stroke/TIA, pacemaker/arrhythmia",
        "- Respiratory disease: asthma, COPD, obstructive sleep apnoea",
        "- Diabetes and wound-healing concerns",
        "- Kidney disease, liver disease, significant blood disorders",
        "- Previous joint replacement, prosthetic joint infection, current skin infections or ulcers",
        "- Restricted neck/jaw movement if regional/general anaesthesia may be considered",
        "- Functional capacity and ability to climb stairs",
        "",
        "Notable medications to ask about",
        "- Anticoagulants and antiplatelets: warfarin, DOACs, heparin, aspirin, clopidogrel, ticagrelor",
        "- Steroids or immunosuppressants",
        "- Diabetes medicines including insulin",
        "- Regular opioids, gabapentinoids, NSAIDs, chronic pain medicines",
        "- Osteoporosis medicines where relevant",
        "- More than 3 prescribed medicines",
        "",
        "Allergies / reactions to ask about",
        "- Metal allergy, latex, plasters/adhesives, medicines, antiseptics",
        "- Previous anaphylaxis",
        "",
        "Other relevant chatbot prompts",
        "- Ask about mobility aids, transport needs, home setup, stairs, and discharge support.",
        "- Ask about current wounds, infections, dental infections, or recent antibiotics.",
        "- Ask about previous anaesthetic problems and regional anaesthetic/spinal issues where relevant."
    ), $char(10))
}

/// Urology — inlined from specialty_guides_txt/ST-UROL_urology.txt.
ClassMethod UrolContent() As %String
{
    Quit $listtostring($listbuild(
        "Specialty ID: ST-UROL",
        "Specialty: Urology",
        "",
        "Purpose",
        "Synthetic chatbot screening guide for patients scheduled for urological procedures.",
        "",
        "Relevant procedures in dataset",
        "- Transurethral resection of prostate",
        "- Ureteroscopy and laser lithotripsy",
        "- Circumcision",
        "- Nephrectomy",
        "- Cystoscopy with bladder biopsy",
        "",
        "Risk factors to ask about",
        "- Kidney disease, urinary retention, recurrent urinary tract infections, bladder problems",
        "- Diabetes and infection risk",
        "- Heart disease, hypertension, stroke/TIA, DVT/PE, bleeding disorders",
        "- Respiratory disease: asthma, COPD, obstructive sleep apnoea",
        "- Liver disease, jaundice, hepatitis",
        "- Current infection symptoms: fever, dysuria, flank pain, worsening urinary symptoms",
        "- Previous anaesthetic complications",
        "- Functional capacity, mobility limitations, transport needs",
        "",
        "Notable medications to ask about",
        "- Anticoagulants and antiplatelets: warfarin, DOACs, heparin, aspirin, clopidogrel, ticagrelor",
        "- Alpha blockers: tamsulosin, alfuzosin, doxazosin",
        "- 5-alpha-reductase inhibitors: finasteride, dutasteride",
        "- Diabetes medicines including insulin and SGLT2 inhibitors",
        "- Antibiotics or recent treatment for UTI",
        "- Steroids or immunosuppressants",
        "",
        "Allergies / reactions to ask about",
        "- Medicines, latex, plasters/adhesives, antiseptics, food allergies",
        "- Previous anaphylaxis",
        "",
        "Other relevant chatbot prompts",
        "- Ask about current urinary infection symptoms and whether a urine sample/culture has been requested.",
        "- Ask about catheters, stents, nephrostomy tubes, or urinary retention.",
        "- Ask about kidney function and whether recent blood tests are available."
    ), $char(10))
}

Storage Default
{
<Data name="SpecialtyGuideDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>SpecialtyID</Value>
</Value>
<Value name="3">
<Value>SpecialtyName</Value>
</Value>
<Value name="4">
<Value>Procedures</Value>
</Value>
<Value name="5">
<Value>Content</Value>
</Value>
<Value name="6">
<Value>CreatedAt</Value>
</Value>
<Value name="7">
<Value>UpdatedAt</Value>
</Value>
</Data>
<DataLocation>^Data.SpecialtyGuideD</DataLocation>
<DefaultData>SpecialtyGuideDefaultData</DefaultData>
<IdLocation>^Data.SpecialtyGuideD</IdLocation>
<IndexLocation>^Data.SpecialtyGuideI</IndexLocation>
<StreamLocation>^Data.SpecialtyGuideS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
