/// A patient's scheduled (or completed) clinical procedure.
/// 
/// One row per booking. `PatientSSN` is indexed so the chatbot can quickly
/// fetch a patient's appointments by SSN, and `SpecialtyID` is the link to
/// `Data.SpecialtyGuide` for pre-op screening.
Class Data.ScheduledProcedure Extends %Persistent
{

/// Patient SSN (9 digits, no separators). Indexed.
Property PatientSSN As %String(MAXLEN = 32) [ Required ];

/// Specialty key — joins to Data.SpecialtyGuide.SpecialtyID (e.g. "ST-ENDO").
Property SpecialtyID As %String(MAXLEN = 32) [ Required ];

/// Procedure name (e.g. "Total thyroidectomy").
Property ProcedureName As %String(MAXLEN = 200) [ Required ];

/// Scheduled date/time of the procedure.
Property ScheduledDate As %TimeStamp;

/// Lifecycle status of the booking.
Property Status As %String(VALUELIST = ",Scheduled,Confirmed,Completed,Cancelled,NoShow") [ InitialExpression = "Scheduled" ];

/// Patient-reported pre-op screening info (free text the chatbot collects).
Property CurrentMedications As %String(MAXLEN = 1000);

Property Allergies As %String(MAXLEN = 1000);

Property RiskFactors As %String(MAXLEN = 2000);

/// Free-form notes (e.g. clinician hand-off summary).
Property Notes As %String(MAXLEN = 2000);

Property CreatedAt As %TimeStamp [ InitialExpression = {$ZDATETIME($HOROLOG, 3)} ];

Property UpdatedAt As %TimeStamp;

/// Patient SSN-driven lookup index — the workflow's primary access path.
Index PatientSSNIndex On PatientSSN;

/// Helps reverse navigation from a specialty back to its bookings.
Index SpecialtyIDIndex On SpecialtyID;

/// Composite for "next procedure for SSN" style queries.
Index PatientDateIndex On (PatientSSN, ScheduledDate);

ClassMethod Populate() As %Status
{
    // Skip seeding if rows already exist — bookings have no unique key,
    // so re-running would create duplicates. SSNs match patients seeded
    // in Data.Patients.Populate so the demo flow ties together.
    &sql(SELECT COUNT(*) INTO :n FROM Data.ScheduledProcedure)
    If (n > 0) { Return $$$OK }
    // One booking per patient seeded in Data.Patients.Populate; every
    // specialty defined in Data.SpecialtyGuide.Populate gets exercised
    // (each specialty appears twice, with a different procedure each time).
    Do ..SaveProcedure("482739156", "ST-ENDO", "Total thyroidectomy",            "2026-05-15 09:00:00")
    Do ..SaveProcedure("593817264", "ST-ORTH", "Total knee replacement",         "2026-05-22 08:30:00")
    Do ..SaveProcedure("671928345", "ST-GENS", "Laparoscopic cholecystectomy",   "2026-06-03 11:00:00")
    Do ..SaveProcedure("728394651", "ST-UROL", "Cystoscopy with bladder biopsy", "2026-06-10 14:00:00")
    Do ..SaveProcedure("839201746", "ST-CARD", "Aortic valve replacement",       "2026-06-18 07:30:00")
    Do ..SaveProcedure("910283746", "ST-ENDO", "Parathyroidectomy",              "2026-06-25 10:00:00")
    Do ..SaveProcedure("104857362", "ST-ORTH", "ACL reconstruction",             "2026-07-02 08:00:00")
    Do ..SaveProcedure("219384756", "ST-GENS", "Open inguinal hernia repair",    "2026-07-09 13:30:00")
    Do ..SaveProcedure("384756192", "ST-UROL", "Ureteroscopy and laser lithotripsy", "2026-07-16 09:30:00")
    Do ..SaveProcedure("475839201", "ST-CARD", "Coronary artery bypass graft",   "2026-07-23 07:00:00")
    Return $$$OK
}

ClassMethod SaveProcedure(PatientSSN As %String, SpecialtyID As %String, ProcedureName As %String, ScheduledDate As %TimeStamp) As %Status
{
    Set procedure = ##class(Data.ScheduledProcedure).%New()
    Set procedure.PatientSSN = PatientSSN
    Set procedure.SpecialtyID = SpecialtyID
    Set procedure.ProcedureName = ProcedureName
    Set procedure.ScheduledDate = ScheduledDate
    Set procedure.Status = "Scheduled"
    Do procedure.%Save()
    Return $$$OK
}

Storage Default
{
<Data name="ScheduledProcedureDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>PatientSSN</Value>
</Value>
<Value name="3">
<Value>SpecialtyID</Value>
</Value>
<Value name="4">
<Value>ProcedureName</Value>
</Value>
<Value name="5">
<Value>ScheduledDate</Value>
</Value>
<Value name="6">
<Value>Status</Value>
</Value>
<Value name="7">
<Value>CurrentMedications</Value>
</Value>
<Value name="8">
<Value>Allergies</Value>
</Value>
<Value name="9">
<Value>RiskFactors</Value>
</Value>
<Value name="10">
<Value>Notes</Value>
</Value>
<Value name="11">
<Value>CreatedAt</Value>
</Value>
<Value name="12">
<Value>UpdatedAt</Value>
</Value>
</Data>
<DataLocation>^Data.ScheduledProcedureD</DataLocation>
<DefaultData>ScheduledProcedureDefaultData</DefaultData>
<IdLocation>^Data.ScheduledProcedureD</IdLocation>
<IndexLocation>^Data.ScheduledProcedureI</IndexLocation>
<StreamLocation>^Data.ScheduledProcedureS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
