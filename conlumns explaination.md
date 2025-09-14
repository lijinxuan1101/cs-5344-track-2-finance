## Column Names and Descriptions of the Freddie Mac Single-Family Loan-Level Dataset

**index**  
Unique identifier assigned to each loan.

**target**  
Binary label indicating loan performance outcome (for this project setting):

- `0` = normal loan (no default).
- `1` = abnormal loan (default or anomalous event).

---

### Origination Variables

- **CreditScore**  
  Borrower credit score at origination (300–850). Values outside range or missing coded as `9999`.

- **FirstPaymentDate**  
  First scheduled payment due date (YYYYMM).
  - convert into pd.datetime

- **FirstTimeHomebuyerFlag**  
  `Y` = Yes, `N` = No, `9` = Not Available.

- **MaturityDate**  
  Scheduled maturity date (YYYYMM).
  - convert into pd.datetime

- **MSA**  
  Metropolitan Statistical Area code (null if unknown).

- **MI_Pct**  
  Mortgage insurance percentage. `0` = none, `1–55` = coverage %, `999` = not available.

- **NumberOfUnits**  
  Number of dwelling units (1–4).

- **OccupancyStatus**  
  `P` = Primary, `I` = Investment, `S` = Second Home, `9` = Not Available.

- **OriginalCLTV**  
  Combined Loan-to-Value ratio at origination.

- **OriginalDTI**  
  Debt-to-Income ratio (%). Values > 65% or missing coded as `999`.

- **OriginalUPB**  
  Original unpaid principal balance (nearest \$1,000).

- **OriginalLTV**  
  Loan-to-Value ratio at origination; invalid coded as `999`.

- **OriginalInterestRate**  
  Note rate at origination.

- **Channel**  
  Origination channel: `R` = Retail, `B` = Broker, `C` = Correspondent, `T` = TPO Not Specified, `9` = Not Available.

- **PPM_Flag**  
  Prepayment penalty: `Y` = Yes, `N` = No.

- **ProductType**  
  `FRM` = Fixed Rate, `ARM` = Adjustable Rate.

- **PropertyState**  
  Two-letter state/territory code.

- **PropertyType**  
  `SF` = Single-Family, `CO` = Condo, `PU` = PUD, `MH` = Manufactured, `CP` = Co-op, `99` = Not Available.

- **PostalCode**  
  Masked ZIP code (first 3 digits + “00”).

- **LoanPurpose**  
  `P` = Purchase, `C` = Refinance Cash Out, `N` = Refinance No Cash Out, `R` = Refinance Not Specified, `9` = Not Available.

- **OriginalLoanTerm**  
  Scheduled term in months.

- **NumberOfBorrowers**  
  Number of borrowers (1–10).

- **SellerName**  
  Entity that sold the loan (“Other Sellers” if below disclosure threshold).

- **ServicerName**  
  Entity servicing the loan (“Other Servicers” if below disclosure threshold).

- **SuperConformingFlag**  
  Indicates whether loan exceeded conforming limits but qualified as “super conforming”.

- **PreHARP_Flag / ProgramIndicator / ReliefRefinanceIndicator**  
  Indicators for HARP and related refinance programs.
  - dropped PreHARP_Flag (missing rate 100%)
  - dropped ReliefRefinanceIndicator(missing rate 100%)
- **PropertyValMethod**  
  Appraisal method: `1` = ACE, `2` = Full, `3` = Other (Desktop/AVM), `4` = ACE + PDR.

- **InterestOnlyFlag**  
  `Y` = interest-only payments required, else `N`.

- **BalloonIndicator**  
  `Y` = balloon payment, else `N`.

---

### Performance Panel Variables

For each loan, monthly performance data is provided across multiple periods.  
The prefix **N_** indicates the month index, where `N = 0, 1, 2, …`.  
Each panel contains the following repeated fields:

- **N_CurrentActualUPB**  
  Current unpaid principal balance (UPB), including both interest-bearing and non-interest-bearing portions.

- **N_CurrentInterestRate**  
  Mortgage interest rate in effect for that period.

- **N_CurrentNonInterestBearingUPB**  
  Non-interest-bearing portion of UPB (e.g., deferred modification amounts).

- **N_EstimatedLTV**  
  Current estimated Loan-to-Value ratio (ELTV) from Freddie Mac’s AVM.  
  Range: 1–998, with `999` = unknown.

- **N_InterestBearingUPB**  
  Portion of UPB that accrues interest.

- **N_LoanAge**  
  Number of months since the loan’s first payment date (or modification date).

- **N_MonthlyReportingPeriod**  
  Period identifier in YYYYMM format.
  - convert into pd.datetime

- **N_RemainingMonthsToLegalMaturity**  
  Remaining months until scheduled maturity (adjusted if modified).

---

### Notes

- The origination variables provide static background (borrower credit, loan terms, property information).
- The performance panel makes this a longitudinal dataset: each loan is tracked monthly until payoff, maturity, or default.
- For further detail, see the official Freddie Mac user guide.