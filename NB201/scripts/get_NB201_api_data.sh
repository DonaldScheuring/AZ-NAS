## Implementation of AZ-NAS on NAS-Bench-201 (NATS-Bench-TSS)
#- Prepare the API file from [NATS-Bench](https://github.com/D-X-Y/NATS-Bench/tree/1d4a304ad1906aa5866563438fcbf0d624b7eda2) (e.g., `./api_data/NATS-tss-v1_0-3ffb9-simple`).
#- Run `tss_general.ipynb` for experiments.

#https://drive.google.com/file/d/17_saCsj_krKjlCBLOJEpNtzPXArMCqxU/view?usp=drive_link
#id: 17_saCsj_krKjlCBLOJEpNtzPXArMCqxU

mkdir -p api_data

gdown --id 17_saCsj_krKjlCBLOJEpNtzPXArMCqxU -O api_data/NATS-tss-v1_0-3ffb9-simple.tar

tar -xvf api_data/NATS-tss-v1_0-3ffb9-simple.tar -C api_data/
