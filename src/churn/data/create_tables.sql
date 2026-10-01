-- queries to create the sql tables from csv files 

CREATE TABLE Labels (
    MsNo CHAR(44) PRIMARY KEY,
    IsChurn BOOLEAN NOT NULL
);

CREATE TABLE Transactions (
    MsNo CHAR(44), 
    PaymentMethodID VARCHAR(10), 
    PaymentPlanDays VARCHAR(10), 
    PlanListPrice VARCHAR(10), 
    ActualAmountPaid VARCHAR(10), 
    IsAutoRenew BOOLEAN, 
    TransactionDate INT,
    MembershipExpireDate INT, 
    IsCancel BOOLEAN
);


CREATE TABLE UserLogs (
    MsNo CHAR(44), 
    LogDate INT, 
    Num25 INT, 
    Num50 INT, 
    Num75 INT, 
    Num985 INT,
    Num100 INT,
    NumUnq INT, 
    TotalSecs FLOAT 
);

--- members_v3

CREATE TABLE Members (
    MsNo CHAR(44) PRIMARY KEY, 
    City INT, 
    BD INT, 
    Gender VARCHAR(10), 
    RegisteredVia INT, 
    RegistrationInitDate INT
);

