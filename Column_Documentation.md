## Freddie Mac Single-Family Loan Dataset Column Names and Descriptions

**index**
A unique identifier assigned to each loan.

**target** Binary classification label indicating loan performance outcome (project-specific):

- `0` = Normal loan (no default).
- `1` = Anomalous loan (default or anomalous event).

---

### Origination Variables

- **CreditScore** Borrower credit score at loan origination (300-850). Values outside range or missing are encoded as `9999`.
- **FirstPaymentDate** First scheduled payment date (YYYYMM format).

  - Converted to pd.datetime
- **FirstTimeHomebuyerFlag** `Y` = Yes, `N` = No, `9` = Not available.

  - One-hot encoded
  - Mapping: `FirstTimeHomebuyerFlag` -> `FirstTimeHomebuyerFlag_N`, `FirstTimeHomebuyerFlag_Y`
- **MaturityDate** Scheduled maturity date (YYYYMM format).

  - Converted to pd.datetime
- **MSA** Metropolitan Statistical Area code (null when unknown).
- **MI_Pct** Mortgage insurance percentage. `0` = None, `1–55` = Coverage %, `999` = Not available.
- **NumberOfUnits** Number of housing units (1-4).
- **OccupancyStatus** `P` = Primary residence, `I` = Investment, `S` = Second home, `9` = Not available.

  - One-hot encoded
  - Mapping: `OccupancyStatus` -> `OccupancyStatus_I`, `OccupancyStatus_P`, `OccupancyStatus_S`
- **OriginalCLTV** Combined Loan-to-Value ratio at loan origination.
- **OriginalDTI** Debt-to-Income ratio (%). Values greater than 65% or missing are encoded as `999`.
- **OriginalUPB** Original unpaid principal balance (nearest $1,000).
- **OriginalLTV** Loan-to-Value ratio at loan origination; invalid values encoded as `999`.
- **OriginalInterestRate** Note rate at loan origination.
- **Channel** Loan channel: `R` = Retail, `B` = Broker, `C` = Correspondent, `T` = Third-party originator unspecified, `9` = Not available.

  - One-hot encoded
  - Mapping: `Channel` -> `Channel_B`, `Channel_C`, `Channel_R`
- **PPM_Flag** Prepayment penalty: `Y` = Yes, `N` = No.

  - One-hot encoded
  - Mapping: `PPM_Flag` -> `PPM_Flag_N`
- **ProductType** `FRM` = Fixed rate, `ARM` = Adjustable rate.

  - One-hot encoded
  - Mapping: `ProductType` -> `ProductType_FRM`
- **PropertyState** Two-letter state/territory code.
- **PropertyType** `SF` = Single-family, `CO` = Condo, `PU` = Planned unit development, `MH` = Manufactured housing, `CP` = Co-op, `99` = Not available.

  - One-hot encoded
  - Mapping: `PropertyType` -> `PropertyType_CO`, `PropertyType_CP`, `PropertyType_MH`, `PropertyType_PU`, `PropertyType_SF`
- **PostalCode** Masked ZIP code (first 3 digits + "00").
- **LoanPurpose** `P` = Purchase, `C` = Cash-out refinance, `N` = No cash-out refinance, `R` = Refinance unspecified, `9` = Not available.

  - One-hot encoded
  - Mapping: `LoanPurpose` -> `LoanPurpose_C`, `LoanPurpose_N`, `LoanPurpose_P`
- **OriginalLoanTerm** Scheduled term (months).
- **NumberOfBorrowers** Number of borrowers (1-10).
- **SellerName** Entity that sold the loan ("Other Seller" when below disclosure threshold).
- **ServicerName** Entity that services the loan ("Other Servicer" when below disclosure threshold).
- **SuperConformingFlag** Indicates whether loan exceeds conforming loan limit but qualifies as "super-conforming".

  - One-hot encoded
  - Mapping: `SuperConformingFlag` -> `SuperConformingFlag_Y`
- **PreHARP_Flag / ProgramIndicator / ReliefRefinanceIndicator** Indicators for HARP and related refinance programs.

  - Removed PreHARP_Flag (100% missing rate)
  - Removed ReliefRefinanceIndicator (100% missing rate)
  - ProgramIndicator: One-hot encoded
  - Mapping: `ProgramIndicator` -> `ProgramIndicator_9`, `ProgramIndicator_F`, `ProgramIndicator_H`
- **PropertyValMethod** Valuation method: `1` = ACE, `2` = Full appraisal, `3` = Other (desktop appraisal/AVM), `4` = ACE + PDR.
- **InterestOnlyFlag** `Y` = Interest-only required, otherwise `N`.

  - One-hot encoded
  - Mapping: `InterestOnlyFlag` -> `InterestOnlyFlag_N`
- **BalloonIndicator** `Y` = Balloon loan, otherwise `N`.

  - One-hot encoded
  - Mapping: `BalloonIndicator` -> `BalloonIndicator_7`, `BalloonIndicator_N`, `BalloonIndicator_Y`

---

### Performance Panel Variables

For each loan, monthly performance data is provided across multiple periods. The prefix **N_** indicates month index, where `N = 0, 1, 2, …`. Each panel contains the following repeating fields:

- **N_CurrentActualUPB** Current unpaid principal balance (UPB), including both interest-bearing and non-interest-bearing portions.
- **N_CurrentInterestRate** Mortgage interest rate effective during the period.
- **N_CurrentNonInterestBearingUPB** Non-interest-bearing portion of UPB (e.g., deferred modification amounts).
- **N_EstimatedLTV** Current estimated loan-to-value ratio (ELTV) from Freddie Mac's Automated Valuation Model (AVM). Range: 1-998, `999` = Unknown.
- **N_InterestBearingUPB** Interest-bearing portion of UPB.
- **N_LoanAge** Number of months since first payment date (or modification date).
- **N_MonthlyReportingPeriod** Period identifier (YYYYMM format).

  - Converted to pd.datetime
- **N_RemainingMonthsToLegalMaturity**
  Remaining months to scheduled maturity (adjusted if modified).

---

### Notes

- Origination variables provide static background information (borrower credit, loan terms, property information).
- Performance panels make this a longitudinal dataset: each loan is tracked monthly until paid off, matured, or defaulted.
- For more detailed information, refer to the official Freddie Mac user guide.

