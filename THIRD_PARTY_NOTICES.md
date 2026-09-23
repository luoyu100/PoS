# Third-party notices

The root MIT license covers original PoS code only. It does not grant rights to
benchmark datasets, model services, downloaded packages, or third-party sources.
Keep upstream attribution and license files when preparing or redistributing data.

| Component | Source and terms | Release treatment |
| --- | --- | --- |
| ALFWorld | [Official repository](https://github.com/alfworld/alfworld), upstream software and asset terms | Installed as an optional dependency; evaluation files downloaded separately |
| LOCA-Bench | [Official repository](https://github.com/hkust-nlp/LOCA-bench), MIT | Runtime included directly; notice retained in `benchmarks/LOCA-Bench/LICENSE` |
| GEM files within LOCA | Individual files credit the AxonRL Team under Apache-2.0 | Original headers preserved; full license in `benchmarks/LOCA-Bench/LICENSE-APACHE-2.0` |
| RCA100 | [Public data root](https://aiops-benchmark.oss-cn-hongkong.aliyuncs.com/rca/rca100/v1.1/), CC BY-NC-SA 4.0 for the described dataset | License retained in `benchmarks/RCA100/LICENSE`; evaluator key has additional distribution instructions in its own README |
| ClinDiag | [Official repository](https://github.com/geteff1/ClinDiag), upstream access and data terms | User-authorized archive import; no clinical records bundled |
| MCP tools and Python/Node dependencies | Their respective upstream projects | Installed separately; not relicensed by PoS |

The included LOCA adaptations change executable selection for local tools and
the active Python interpreter. They do not replace upstream copyright notices
or modify benchmark scoring.

RCA100 evaluator labels are not part of the public observation download. They
are bundled in this release with public redistribution authorization confirmed
by the release maintainer. The original answer-key README and attribution are
preserved, with the personal contact email omitted for anonymous review. This
inclusion does not relicense the labels under MIT or
remove applicable third-party conditions.

Third-party names appearing here identify sources and rights holders, not the
identity of the anonymous PoS contributors. No original experiment outputs,
private model credentials, or generated benchmark workspaces are distributed.
