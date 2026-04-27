/// Patient master record used by the admission chatbot.
/// SSN is the natural unique identifier (one row per person).
Class Data.Patients Extends %Persistent
{

/// Social Security Number — unique business key for the patient.
Property SSN As %String(MAXLEN = 32) [ Required ];

Property FirstName As %String(MAXLEN = 100) [ Required ];

Property LastName As %String(MAXLEN = 100) [ Required ];

Property DateOfBirth As %String;

Property Gender As %String(VALUELIST = ",M,F,Other,Unknown");

Property TelephoneNumber As %String(MAXLEN = 32);

Property Address As %String(MAXLEN = 255);

/// Audit columns
Property CreatedAt As %TimeStamp [ InitialExpression = {$ZDATETIME($HOROLOG, 3)} ];

Property UpdatedAt As %TimeStamp;

/// SSN must be unique across patients.
Index SSNIndex On SSN [ PrimaryKey, Unique ];

/// Helps name-based lookups during admission.
Index NameIndex On (LastName, FirstName);

ClassMethod Populate() As %Status
{
    Set sc = ..SavePatient("482739156", "John", "Doe", "1985-03-12", "M", "", "742 Evergreen Terrace, Springfield, IL")
    $$$ThrowOnError(sc)
    Do ..SavePatient("593817264", "Emily", "Carter", "1992-07-25", "", "", "123 Maple Street, Denver, CO")
    Do ..SavePatient("671928345", "Michael", "Hudson", "1978-11-03", "M", "", "88 Market Street, San Francisco, CA")
    Do ..SavePatient("728394651", "Sophia", "Rossi", "1995-01-17", "", "", "15 Broadway, New York, NY")
    Do ..SavePatient("839201746", "Liam", "O'Connor", "1988-09-09", "M", "617-555-0134", "")
    Do ..SavePatient("910283746", "Ava", "Martinez", "2000-05-30", "F", "", "200 Ocean Drive, Miami, FL")
    Do ..SavePatient("104857362", "Noah", "Schmidt", "1983-12-21", "", "312-555-0167", "")
    Do ..SavePatient("219384756", "Isabella", "Kowalski", "1997-08-14", "F", "", "")
    Do ..SavePatient("384756192", "Ethan", "Dubois", "1990-04-02", "", "", "600 Canal Street, New Orleans, LA")
    Do ..SavePatient("475839201", "Mia", "Haddad", "1986-06-18", "F", "602-555-0159", "")
    Return $$$OK
}

ClassMethod SavePatient(SSN As %String, FirstName As %String, LastName As %String, DateOfBirth As %Date = "", Gender As %String = "", TelephoneNumber As %String = "", Address As %String = "") As %Status
{
    Set patient = ##class(Data.Patients).%New()
    Set patient.SSN = SSN
    Set patient.FirstName = FirstName
    Set patient.LastName = LastName
    Set patient.DateOfBirth = DateOfBirth
    Set patient.Gender = Gender
    Set patient.TelephoneNumber = TelephoneNumber
    Set patient.Address = Address
    Set sc = patient.%Save()
    If $$$ISERR(sc) {
        Return sc
    }
    Return $$$OK
}

Storage Default
{
<Data name="PatientsDefaultData">
<Value name="1">
<Value>%%CLASSNAME</Value>
</Value>
<Value name="2">
<Value>SSN</Value>
</Value>
<Value name="3">
<Value>FirstName</Value>
</Value>
<Value name="4">
<Value>LastName</Value>
</Value>
<Value name="5">
<Value>DateOfBirth</Value>
</Value>
<Value name="6">
<Value>Gender</Value>
</Value>
<Value name="7">
<Value>TelephoneNumber</Value>
</Value>
<Value name="8">
<Value>Address</Value>
</Value>
<Value name="9">
<Value>CreatedAt</Value>
</Value>
<Value name="10">
<Value>UpdatedAt</Value>
</Value>
</Data>
<DataLocation>^Data.PatientsD</DataLocation>
<DefaultData>PatientsDefaultData</DefaultData>
<IdLocation>^Data.PatientsD</IdLocation>
<IndexLocation>^Data.PatientsI</IndexLocation>
<StreamLocation>^Data.PatientsS</StreamLocation>
<Type>%Storage.Persistent</Type>
}

}
