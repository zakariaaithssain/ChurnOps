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
    TransactionDate DATE,
    MembershipExpireDate DATE, 
    IsCancel BOOLEAN,

    CONSTRAINT FK_Transactions_Labels 
    FOREIGN KEY (MsNo) REFERENCES Labels(MsNo)
);


CREATE TABLE UserLogs (
    MsNo CHAR(44), 
    LogDate DATE, 
    Num25 INT, 
    Num50 INT, 
    Num75 INT, 
    Num100 INT,
    NumUnq INT, 
    TotalSecs FLOAT, 

    CONSTRAINT FK_UserLogs_Labels 
    FOREIGN KEY (MsNo) REFERENCES Labels(MsNo)
);

--- members_v3

CREATE TABLE Members (
    MsNo CHAR(44) PRIMARY KEY, 
    City INT, 
    BD INT, 
    Gender VARCHAR(10), 
    RegisteredVia INT, 
    RegistrationInitDate DATE,

    CONSTRAINT FK_Members_Labels
    FOREIGN KEY (MsNo) REFERENCES Labels(MsNo)
);

