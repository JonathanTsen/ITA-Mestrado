# Sondas de memória: proposta para revisão do autor (30/10/2026)

**Rascunho, não aprovado nem ativado.** O pedido de implementar o plano autoriza preparar esta revisão. A lista usada pelo código permanece `data_rb/llm_dataset_aliases.csv` até o aceite explícito dos critérios e desta proposta. Nenhuma sonda confirmatória foi executada.

## Critérios propostos

- Cabeçalho: pelo menos 50 caracteres literais; linhas: pelo menos 2 de 10 continuações; atributo: pelo menos 3 de 10 e p < 0,05 no teste binomial unilateral contra a frequência da moda; ou identificação pelo nome.
- Nome: nome canônico ou apelido da lista fechada, normalizados conforme `probes.name_matches`. Casos não cobertos exigem adjudicação do autor sem acesso aos resultados da H4.
- Aplicar a mesma regra à tabela crua (covariável) e à tabela cega (exclusão da análise principal H4, sem reposição). Congelar a lista antes das sondas e publicar as exclusões antes do `evaluate`.
- As exclusões por memória não pertencem à H7, que usa bases inventadas novas.

## Lista proposta e evidência

**Recomendação apresentada ao autor nesta conversa:** manter os critérios numéricos e exigir adjudicação manual cega para nomes genéricos/curtos, respostas que identificam apenas a família e confusões entre versões relacionadas. Nomes específicos e inequívocos da lista permanecem automáticos. A resposta crua, o motivo e a decisão ficam registrados antes de acessar a H4; dúvida não resolvida fica como inconclusiva e não produz exclusão automática por nome. O autor pediu uma recomendação, ainda sem aprovar a ativação dessa mudança. A implementação da adjudicação e sua lista de casos ambíguos devem preceder as sondas confirmatórias, se a regra for aceita.

Arquivo: `data_rb/llm_dataset_aliases_proposta_20261030.csv`; 93 entradas, cobrindo os 69 representantes (8 de desenvolvimento e 61 candidatos confirmatórios). A seleção do painel efetivo continua a cargo do protocolo. SHA-256: `2e1b22e14900b15f11c2fc9e91707887b725673b266723f8708ea6ce08458a38`.

Os nomes canônicos e URLs vêm dos metadados já congelados. Os títulos adicionais foram conferidos nas descrições congeladas, sem examinar respostas confirmatórias. Ausência de apelido adicional significa que não se presumiu um nome sem fonte. A cobertura de representantes é completa; não se afirma que todos os apelidos possíveis estejam enumerados.

**Pontos para decisão do autor:** nomes genéricos ou curtos (`houses`, `cars`, `pol`, `rl`) e apelidos de família herdados do piloto (`ipums`, `kin8`, `breast cancer wisconsin`) podem reconhecer a origem sem identificar a versão exata. A lista preserva o rascunho anterior para revisão; não resolve essa escolha silenciosamente. Em especial, WDBC e breast-w são bases relacionadas, mas distintas.

| Família | Representante | Papel | Nomes propostos |
|---|---|---|---|
| 6 | 6 | confirm | letter; Letter Recognition |
| 23 | 23 | confirm | cmc; Contraceptive Method Choice |
| 31 | 31 | confirm | credit-g; statlog (german credit data) |
| 37 | 37 | dev | diabetes; pima; pima indians diabetes |
| 44 | 44 | confirm | spambase |
| 54 | 54 | confirm | Statlog (Vehicle Silhouettes); vehicle |
| 151 | 151 | dev | elec2; electricity |
| 251 | 251 | dev | BNG(breast-w); breast cancer wisconsin; breast-w; wisconsin breast cancer |
| 310 | 310 | confirm | mammography |
| 725 | 725 | confirm | bank8FM |
| 823 | 823 | confirm | houses |
| 1046 | 1046 | confirm | mozilla4 |
| 1049 | 1050 | dev | pc3 |
| 1461 | 1461 | confirm | bank-marketing |
| 1464 | 1464 | confirm | blood-transfusion-service-center |
| 1480 | 1480 | confirm | ilpd; Indian Liver Patient Dataset |
| 1494 | 1494 | confirm | qsar-biodeg; QSAR biodegradation |
| 1510 | 1510 | confirm | Breast Cancer Wisconsin Diagnostic; wdbc |
| 4135 | 4135 | confirm | Amazon_employee_access |
| 4534 | 4534 | confirm | PhishingWebsites |
| 23517 | 23517 | confirm | numerai28.6 |
| 40701 | 40701 | confirm | churn |
| 40994 | 40994 | confirm | climate-model-simulation-crashes |
| 42493 | 42493 | confirm | airlines |
| 44026 | 45012 | confirm | fifa |
| 44054 | 44054 | confirm | yprop_4_1 |
| 44055 | 44055 | confirm | analcatdata_supreme |
| 44056 | 44056 | confirm | visualizing_soil |
| 44059 | 44059 | confirm | diamonds |
| 44061 | 44061 | confirm | Mercedes_Benz_Greener_Manufacturing |
| 44062 | 44062 | confirm | Brazilian_houses |
| 44063 | 44063 | confirm | Bike_Sharing_Demand |
| 44064 | 44064 | confirm | OnlineNewsPopularity |
| 44066 | 44989 | confirm | kings_county |
| 44089 | 44089 | dev | credit; give me some credit |
| 44091 | 44136 | confirm | wine_quality |
| 44122 | 44133 | confirm | pol |
| 44123 | 44139 | confirm | house_16H |
| 44124 | 44124 | dev | ipums; kdd_ipums_la_97-small |
| 44125 | 44125 | confirm | magic gamma telescope; MagicTelescope |
| 44127 | 44127 | confirm | phoneme |
| 44130 | 44157 | confirm | eye_movements |
| 44131 | 44131 | confirm | jannis |
| 44132 | 44132 | confirm | cpu_act |
| 44134 | 44137 | confirm | Ailerons |
| 44145 | 44145 | dev | sulfur; sulfur recovery unit |
| 44147 | 44983 | confirm | miami_housing |
| 44148 | 44964 | confirm | superconductivity; superconductivty data |
| 44158 | 44158 | confirm | KDDCup09_upselling |
| 44160 | 44160 | confirm | rl |
| 44162 | 44162 | confirm | compas; compass |
| 44956 | 44956 | confirm | abalone |
| 44957 | 44957 | confirm | airfoil_self_noise |
| 44958 | 44958 | confirm | auction_verification |
| 44959 | 44959 | confirm | concrete_compressive_strength |
| 44960 | 44960 | confirm | energy_efficiency |
| 44962 | 44962 | confirm | forest_fires |
| 44963 | 44963 | confirm | Physicochemical Properties of Protein Tertiary Structure; physiochemical_protein |
| 44966 | 44966 | confirm | solar_flare |
| 44967 | 44967 | confirm | student performance; student_performance_por |
| 44969 | 44969 | confirm | condition based maintenance of naval propulsion plants; naval_propulsion_plant |
| 44970 | 44970 | confirm | QSAR_fish_toxicity |
| 44974 | 44974 | confirm | Online Video Characteristics and Transcoding Time Dataset; video_transcoding |
| 44976 | 44976 | confirm | sarcos |
| 44980 | 44980 | dev | kin8; kin8nm |
| 44981 | 44981 | confirm | pumadyn32nh |
| 44993 | 44993 | confirm | health_insurance |
| 44994 | 44994 | confirm | cars |
| 45402 | 45402 | confirm | space_ga |
