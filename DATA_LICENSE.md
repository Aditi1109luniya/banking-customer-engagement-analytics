# Dataset attribution and licensing

The original project code is licensed under [MIT](LICENSE). That license does not replace the source dataset's license.

**Dataset:** Bank Marketing, specifically `bank-additional-full.csv` and its accompanying `bank-additional-names.txt`.

**Creators:** Sérgio Moro, Paulo Rita, and Paulo Cortez.

**Repository citation:** Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306).

**Research citation requested by the additional dataset readme:** Moro, S., Cortez, P., & Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing*. Decision Support Systems. [https://doi.org/10.1016/j.dss.2014.03.001](https://doi.org/10.1016/j.dss.2014.03.001).

**Source:** [UCI Bank Marketing dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing), [official source archive](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip).

UCI lists the dataset under the [Creative Commons Attribution 4.0 International license](https://creativecommons.org/licenses/by/4.0/) (CC BY 4.0). Credit the creators and UCI, link the license, and indicate changes when redistributing adapted data.

This project preserves original source row order, adds surrogate record IDs, normalizes categorical text, converts the subscription label to 0/1, adds fixed descriptive age groups and chronological split metadata, and calculates predictions and analysis summaries. The public demo contains a SQLite representation and a later-period replay subset of that source data. They remain attributed to the original creators under CC BY 4.0. Simulated engagement flags and synthetic input examples are created by this project and clearly labelled.

The source describes historical telephone campaigns at a Portuguese banking institution. It is not ICICI Bank customer data. This project has no bank affiliation or endorsement. No employer campus deck, logo, or private transcript is included in the public codebase.

Exact source bytes, acquisition method, archive member names, and SHA-256 are recorded by `fetch-data` in `data/raw/provenance.json`. The reproducible prepared/demo manifest repeats the source checksum; generated data is not silently substituted for the additional-full variant.
