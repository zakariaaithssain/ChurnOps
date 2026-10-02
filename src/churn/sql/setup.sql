CREATE SCHEMA IF NOT EXISTS round1;
CREATE SCHEMA IF NOT EXISTS round2;

-- members_v3.csv is shared by both rounds
CREATE TABLE IF NOT EXISTS public.members (
    msno                  CHAR(44) PRIMARY KEY,
    city                  INT,
    bd                    INT,            
    gender                VARCHAR(10),   
    registeredvia         INT,
    registrationinitdate  DATE
);


