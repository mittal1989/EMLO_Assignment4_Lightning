# Lightning Template

```
copper_train --help
```

examples

- `copper_train data.num_workers=16`
- `copper_train data.num_workers=16 trainer.deterministic=True +trainer.fast_dev_run=True`

## Development

Install in dev mode

```
pip install -e .
```

### Docker

- `docker build . -t lightning:latest`
- `docker run --name lightning_container lightning:latest copper_train data.num_workers=16 `\
- `docker start lightning_container`
- `docker exec lightning_container copper_eval data.num_workers=16 `

### Team
- `Anurag Mittal`
- `Aman Jaipuria`