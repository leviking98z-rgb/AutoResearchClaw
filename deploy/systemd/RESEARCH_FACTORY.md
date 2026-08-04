# Continuous Research Factory service

The Factory has its own service and configuration. It does **not** replace,
restart, or hot-switch an existing RSI supervisor.

1. Copy `config.factory.example.yaml` to a production-owned path and set
   `factory.enabled: true`.
2. Keep `factory.gpu.claim_on_start: false` when adopting a pool already owned
   by an external allocation lifecycle. Set it to `true` only when the Factory
   service is explicitly designated as that lifecycle owner.
3. Create `/etc/researchclaw-factory/<instance>.env` with mode `0600`:

   ```text
   FACTORY_CONFIG=/absolute/path/to/config.factory.yaml
   ```

4. Install `researchclaw-factory@.service`, run `systemctl daemon-reload`, then
   start a **new** Factory instance:

   ```bash
   systemctl start researchclaw-factory@rsi.service
   journalctl -fu researchclaw-factory@rsi.service
   ```

Control commands are cooperative:

```bash
bin/researchclaw-factory -c /path/config.factory.yaml status
bin/researchclaw-factory -c /path/config.factory.yaml pause maintenance
bin/researchclaw-factory -c /path/config.factory.yaml resume
bin/researchclaw-factory -c /path/config.factory.yaml stop
```

The durable Factory state directory is the source of truth. A service restart
reconciles existing Ideas, Work Items, leases, and detached pool task IDs.
