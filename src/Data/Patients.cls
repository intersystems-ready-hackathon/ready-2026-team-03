/// Patient master record used by the admission chatbot.
/// SSN is the natural unique identifier (one row per person).
Class Data.Patients Extends %Persistent
{

/// Social Security Number — unique business key for the patient.
Property SSN As %String(MAXLEN = 32) [ Required ];

Property FirstName As %String(MAXLEN = 100) [ Required ];

Property LastName As %String(MAXLEN = 100) [ Required ];

Property DateOfBirth As %Date;

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
