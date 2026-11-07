# Segundo depósito do pré-registro (emenda P4): SHA-256 das saídas do run pré-registrado

Data: 06/11/2026. Ele está ligado ao primeiro depósito no Zenodo (conceito `10.5281/zenodo.23933349`; `prereg-v1` = `10.5281/zenodo.23933350`, de 03/11/2026 23:36 UTC).

- **Código que produziu as saídas:** tag `prereg-v1.4` / `prereg-v1.4-llm` (commit `57746cabb2a5bd6fb9df20846b418d76e8b64237`), com a árvore limpa em todos os passos.
- **Run pré-registrado:** `results_rb/20261105-inject-v1-52ed6f93a2a0` (50 famílias, 10.332 instâncias).
- **H7:** `results_rb/20261105-contexto-h7-c635b9dbd73a`.
- **Estudo de caso:** `results_rb/20261106-case-study-f826114aad4f`.
- **Desvios D1–D5:** em `docs/revisao-banca/desvios.md`. O D5 registra que o computador da primeira produção quebrou em 05/11 e que o run completo foi refeito do zero, no mesmo código, numa máquina nova.
- **Os arquivos de resultado não são públicos.** Ficam numa cópia privada (`ITA-dados-privado`). Os SHA-256 abaixo permitem conferir, a qualquer momento, que eles não mudaram depois desta data.

## Resultado da família confirmatória (de `stats_confirmatory.csv`, `stats_llm.csv` e `h7_resultado.json`)

| Hipótese | HL [IC 95%] | Decisão |
|---|---|---|
| H1 | +0,200 [0,179; 0,222] | superior |
| H2 | −0,003 [−0,010; 0,002] | evidência insuficiente |
| H2e | — | equivalentes dentro de ±δ |
| H3 | +0,078 [0,061; 0,095] | superior |
| H5 | +0,085 [0,051; 0,111] | superior |
| H6 | +0,132 [0,112; 0,153] | superior |
| H4 (LLM A1, Argon) | −0,213 [−0,269; −0,167] | inferior |
| H7 (Argon, S7 × S0) | +0,139 [0,083; 0,167] | S7 acima de S0 |

## SHA-256 das saídas (`shasum -a 256`, caminhos relativos a `results_rb/`)

```
8779d8f0dba4db598e2f48a823a30d4a06fab6b413875c0aaa1645b3a5639457  20261105-inject-v1-52ed6f93a2a0/stats_confirmatory.csv
cca587fbb203a3f82c96772a20ceb4811db5cf4d11e2c0b877bf26a1de3e6a38  20261105-inject-v1-52ed6f93a2a0/stats_level.csv
7a4e6199187838655e5691b5271b94a605caf11dfd93d3c85ba75308a71ccffe  20261105-inject-v1-52ed6f93a2a0/stats_h3_decomposition.csv
65364fdc8fe529dd9c6a62621eb4c0ea66b006f2a1541778516d883c717da1d1  20261105-inject-v1-52ed6f93a2a0/stats_h6_auc_by_family.csv
2a454fa2e89887231fb4960770739d8476cad34622709e2a8c9ee3c6962b6a1f  20261105-inject-v1-52ed6f93a2a0/oracle.parquet
d99a51ac418dec2dce481a5dbdf3b364a254b537e552109fdf65ac28a54c43b6  20261105-inject-v1-52ed6f93a2a0/stats_llm.csv
04337d10b78a4df2f5acd59a561dd578855afecd541a018b83ee3714293f857f  20261105-inject-v1-52ed6f93a2a0/stats_llm_covariate.csv
0751743c47a25b71ce9095a0c17740477d47f21fcb349b9960dc9aba3b2d7116  20261105-inject-v1-52ed6f93a2a0/llm-sensitivity/h4_sensitivity.csv
b0581f9f3f23b478597e056a841d5fe2c210be8a58577e75b17bbe7914b4e0e3  20261105-inject-v1-52ed6f93a2a0/llm-a2-ablation/rq4_a2.csv
a18d02c480d74837f9609a18da7d02e45921645298268b9c9a65bb3a5d96ed5d  20261105-inject-v1-52ed6f93a2a0/llm/gemini-4-argon/A1/predictions.parquet
506373f2558c687d960b97e11ceacbef5f832b7dedba517a479bc7e84eb5f17a  20261105-inject-v1-52ed6f93a2a0/llm/gemini-4-argon/probes/exclusions_final.csv
b0be01471e78eba09256454138f7fe8917c97ce44d3a045a1870c3cab44dc291  20261105-inject-v1-52ed6f93a2a0/llm/gemini-4-argon/probes/adjudicacao.csv
045290e208f7d11ca1436adfcd9c5863dabd2fe6ab12f5db898da64d5ff21b4e  20261105-inject-v1-52ed6f93a2a0/llm/gemini-3.8-flash/probes/exclusions_final.csv
9c7158d29a005c240f14b49030dcc9e0071c500c1c3b5068802c05c112646e30  20261105-inject-v1-52ed6f93a2a0/llm/gemini-3.8-flash/probes/adjudicacao.csv
8c9d328d7671831c28f3ad3157fc142ab397cb66e716fa7568521019df635dd3  20261105-inject-v1-52ed6f93a2a0/instances.parquet
0da9ba0e17ac152d3e41f7dcd64cbd5919ea7596d6ee88e48e9187aab47e702c  20261105-inject-v1-52ed6f93a2a0/baselines.parquet
1a86e2434bc66815c2391877508544c1fe17f72d953cffa43f515f001c04e1b2  20261105-inject-v1-52ed6f93a2a0/pklm.parquet
48f2470bff58ccada251481eb33b9aadc5689423252817f6444f3a4a4148697e  20261105-inject-v1-52ed6f93a2a0/run.ablation.json
c62fb072b9357b15a4f2cb9118e259f520a38f5ac49b7aafeb564fc9748b8151  20261105-inject-v1-52ed6f93a2a0/run.baselines-pklm.json
d7ba7318e4cc0d4eff2573161c99c3b006ffa645a56ab1a01e44be854de570c1  20261105-inject-v1-52ed6f93a2a0/run.baselines.json
e518c2192cacab0200aa00376723ccf846eb52fdb4bbc50057ac26e5bec2794d  20261105-inject-v1-52ed6f93a2a0/run.cross-ampute.json
34d041a0a61a14af859202225948daa1b3c32468b74099a1fb57544cfd82fe51  20261105-inject-v1-52ed6f93a2a0/run.cross-generator.json
24ac489ebfba8045f524879a57679b46b15629327bd8c4af675ac310a321846c  20261105-inject-v1-52ed6f93a2a0/run.eval-all.json
d2864f0f91a513caa500e7425f83770fe578a1917d16618992ed31e6f9372474  20261105-inject-v1-52ed6f93a2a0/run.eval-mdatagen.json
b097d6cbb0b06e66669cf08821f0dfd994fac772261c959d6c0a4e27819bb76c  20261105-inject-v1-52ed6f93a2a0/run.eval-symmetric.json
20f4bb811d181f732dc4c4d82ea70cc945c3d1f3882f652aa90b0fcca4afb6ed  20261105-inject-v1-52ed6f93a2a0/run.eval-tab_err.json
939c39400b4a22dc9d2988928e42068c33a0752729fb30a5e8d62013a65a98ca  20261105-inject-v1-52ed6f93a2a0/run.features.json
5172f7933bedfdf43a8524840aafdda3cd967754ff961a364c1a83b2d7177ffb  20261105-inject-v1-52ed6f93a2a0/run.fit-final.json
1a4075b150c6cd4d41b925792502940a2e6030532a2ad0d63ea18c6ec31077f0  20261105-inject-v1-52ed6f93a2a0/run.inject-ampute.json
64095991a1153eb7a7a0769551f2125b130fcac355506506f39df4734a6bce68  20261105-inject-v1-52ed6f93a2a0/run.json
23ed8facab8ec64cb151b66ab86acb997865f3119ed695f2f70688caadbaa73f  20261105-inject-v1-52ed6f93a2a0/run.llm-a2-ablation.json
f28706be53f2ca735ca2c39e119063e05e03074e47565e29269efee1dc47c4c0  20261105-inject-v1-52ed6f93a2a0/run.llm-gemini-4-argon-A1.json
83f01ad9cd355018c0f825abd4affbd6390bf5f7af725275bdd9967e2760fe74  20261105-inject-v1-52ed6f93a2a0/run.llm-gemini-4-argon-A2.json
3b668e7a9284175e4e293ad8f0449fdc1c16ac27bb1514cda91b341633d96efe  20261105-inject-v1-52ed6f93a2a0/run.llm-gemini-4-argon-A3.json
b751fe809368255f1c9b921aff0c03a333f4196005392cfb859726a31b978540  20261105-inject-v1-52ed6f93a2a0/run.llm-probes-gemini-3.8-flash.json
748cad6433ef35fe951eb7281e16d3004191f954a9ce92768a301b079b3a4016  20261105-inject-v1-52ed6f93a2a0/run.llm-probes-gemini-4-argon.json
07996a7cfd8fc509a64a6a7d3b673a9afea9d1aca24f1a7df34c9e54cd4a3cc3  20261105-inject-v1-52ed6f93a2a0/run.llm-sensitivity.json
09c89f2931ad111914b28a7bdb3cd66225b204db3bf49dec87804a7e13831c9b  20261105-inject-v1-52ed6f93a2a0/run.logo.json
0dbf21a9ee0f7aa74c20f8480d69c846e3018b7dc5304f88183108a92296d2ad  20261105-inject-v1-52ed6f93a2a0/run.noise.json
06464255fb6af3c5dd7fc1e26958f2ebe1fe6c92eee07fa29ed0d0b8c9cbb111  20261105-inject-v1-52ed6f93a2a0/run.oracle-features.json
5ee8d953e45b07bbd248c6acde241e96b91594ea53348d4a27802d449a91a005  20261105-inject-v1-52ed6f93a2a0/run.robustness-znan.json
cc906f72f04509eacb16e7c70e62fdb4425029fa8e3b39a9a0453f822ce6f7ed  20261105-inject-v1-52ed6f93a2a0/run.robustness.json
c3770131dbca30bb70c5fa7a7bf68fceb2b591e14fba25e0fdb0d6403f36ae03  20261105-inject-v1-52ed6f93a2a0/run.secondary-extra.json
f5e8147a6a43bae596daedf7debc96fd83e70d0c078db73cfac9c8386ae1077b  20261105-inject-v1-52ed6f93a2a0/run.secondary.json
5e0d6cd2407ca21742a1df12c093ef00498e1e7ffdbf3afa72729b04a5fd022c  20261105-inject-v1-52ed6f93a2a0/run.stats-llm.json
fc34780fef69c560b24e42c07654b9c69b5f95cbf34491d9fb7afb784673e571  20261105-inject-v1-52ed6f93a2a0/run.stats.json
67ab2888ae75cb1cc1ea312f66a8f9cabeae5bbf07d5a147eee975a6d47d885b  20261105-inject-v1-52ed6f93a2a0/run.transfer-20261022-synth-v2-5d31a737645b.json
285e97050dc9cb6a040f29b198578b138bcd2eba1ac9c8a91cf74af8c676c162  20261105-contexto-h7-c635b9dbd73a/analise-gemini-4-argon/h7_resultado.json
e0309660b099ec81d5734550ab208fd1379c50da6c88e210eba01b67025ef7e4  20261105-contexto-h7-c635b9dbd73a/analise-gemini-4-argon/h7_por_base.csv
4b98e2f910d059ee30244ba6e54f90506828668e7e8083c7d6d6f25cc783df19  20261105-contexto-h7-c635b9dbd73a/analise-gemini-3.8-flash/h7_resultado.json
31392f0b53d8f962e83b0d0ece520ba9afed82db6333695d750bace9ae504e95  20261105-contexto-h7-c635b9dbd73a/run.json
2d2f4df6c39af261ba73058ae3bdcfdef41b72abf898705124a446358c123646  20261106-case-study-f826114aad4f/case_summary.csv
5f80279b15162b12739cc0e7de8c16376ab358322e9cb80ef055edcc3535e582  20261106-case-study-f826114aad4f/run.case-study.json
```

## SHA-256 dos documentos do protocolo (neste repositório, no commit deste depósito)

```
715701256e34c653b51b114cb11d9b5eb57e682124762c9a222e6943d884986b  docs/revisao-banca/desvios.md
64a9f0bb3ef2f645fae002bf36da992b6f9683524e78bf1cef734544041199e8  docs/revisao-banca/preregistro.md
ec0fa41b9428c6242174faff75faa1b1c9cdff87dd29334f6b9ff15a95e6c53f  docs/revisao-banca/adendo-llm.md
```

## Aprovação e decisões do autor (repositório da tese `JonathanTsen/ModeloTesePPGPO`, commit `e8f34965cc5e795b9356c4439012de9933b98c4c`)

- `38-preregistro.md`: a aprovação do orientador (29/10/2026, versão `444d1d8`).
- `HANDOFF-2026-11-03.md`: a regra de 03/11 (o autor decide, o orientador revisa os resultados depois) e as decisões do autor.
- `RESULTADOS-2026-11-06*.md`: os resultados oficiais e o anexo, gerados dos arquivos.

```
6b4a85f882df1294108c56b367619868b6059929f98bb7e193ca667a96b7cbe2  plano-revisao-banca/execucao/38-preregistro.md
80a813621a5b3a3420c381ab8d10aa3e8916577f5c21cac2c3d64cd2a7e878d8  plano-revisao-banca/execucao/HANDOFF-2026-11-03.md
951e93101bac3e4a4d8964866aec75b28e3d9a9907c2dc04e325ea1dcb2508f4  plano-revisao-banca/execucao/RESULTADOS-2026-11-06.md
e443a59a7527560de8e52d2d5314c5aba3ddab8ec54075e2ec8d99dd689d72b6  plano-revisao-banca/execucao/RESULTADOS-2026-11-06-ANEXO.md
```
