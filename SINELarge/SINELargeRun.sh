#!/bin/bash
./SINELarge -graph citeseer.txt -output citeseer_SINELarge_emb.txt -time citeseer_SINELarge_time.txt -samples 100
./SINELarge -graph cora.txt -output cora_SINELarge_emb.txt -time cora_SINELarge_time.txt -samples 100
