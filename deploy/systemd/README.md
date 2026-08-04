# AutoResearchClaw RSI systemd operation

These units provide persistent, boot-restorable supervision for an existing
RSI campaign without embedding credentials in unit files.

## Units

- `autoresearch-rsi-supervisor@.service`: resumes the persisted campaign in
  the foreground. The campaign lock prevents duplicate supervisors.
- `autoresearch-rsi-monitor@.service`: polls supervisor, Bridge, allocation,
  GPU, checkpoint, and Ray health. It can invoke `rsi-resume` only when the
  monitor's existing restart policy concludes that the supervisor is unhealthy.
- `autoresearch-rsi-keepalive@.service`: verifies allocation ownership before
  every renewal and never releases an allocation.
- `autoresearch-rsi-dashboard@.service`: serves the campaign-aware local web
  console on `127.0.0.1:8099`. It reads durable campaign state and exposes only
  cooperative pause/resume controls; permanent stop and allocation release are
  intentionally absent.
- `autoresearch-rsi-alert@.service`: sends deduplicated unit-failure notices
  to the configured internal AgentMail recipient.
- `autoresearch-rsi.target`: groups enabled campaign instances for boot.

The supervisor uses `Restart=on-failure`; monitor and allocation keepalive use
`Restart=always` so a clean but unexpected exit cannot silently disable 24/7
coverage. All services wait for network and the relevant mounts and receive
persistent `StateDirectory`, `LogsDirectory`, and runtime directories. Instance
configuration lives in root-readable
`/etc/autoresearch-rsi/<campaign-id>.env` with mode `0600`.

Before resuming a campaign, the supervisor wrapper runs the idempotent
`bin/researchclaw-ensure-deps` check against `RSI_SUPERVISOR_PYTHON`. Missing
small runtime dependencies such as the `arxiv` client are installed into that
exact interpreter; startup fails closed if installation cannot be verified.
Alert routing lives in `/etc/autoresearch-rsi/alerts.env`, also mode `0600`.
It pins the non-secret allocation owner as both
`AUTORESEARCH_CLAIM_OWNER` and `CB_SID`, because a boot-restored service no
longer has an interactive agent parent from which ClusterBridge can infer its
lease identity.

The templates use the campaign ID directly as `%i`; campaign IDs containing
slashes are intentionally unsupported.

## Install without touching production processes

```bash
cd /data/workspace/autoresearch-stack/AutoResearchClaw
sudo deploy/systemd/autoresearch-rsi-install \
  --campaign /root/shared/.clusters/.workdir/autoresearch-rsi/rsi-autonomous-llm-self-improvement \
  --owner 019fc877-7045-7a40-935d-d2bef7883945
```

The installer copies and verifies units and runs `systemctl daemon-reload`. It
does **not** start, stop, restart, or enable anything.

## Integration drill

Do not run these while the legacy production daemons are active. The
`ExecCondition` guards fail closed if they detect an existing supervisor,
monitor, or keepalive.

After a planned cooperative handover:

```bash
INSTANCE=rsi-autonomous-llm-self-improvement

# Inspect the staged definitions first.
sudo systemctl cat "autoresearch-rsi-supervisor@${INSTANCE}.service"
sudo systemctl cat "autoresearch-rsi-monitor@${INSTANCE}.service"
sudo systemctl cat "autoresearch-rsi-keepalive@${INSTANCE}.service"

# Start one component at a time during the drill.
sudo systemctl start "autoresearch-rsi-keepalive@${INSTANCE}.service"
sudo systemctl start "autoresearch-rsi-supervisor@${INSTANCE}.service"
sudo systemctl start "autoresearch-rsi-monitor@${INSTANCE}.service"

# Enable boot restoration only after the live drill succeeds.
sudo systemctl enable autoresearch-rsi.target
sudo systemctl enable \
  "autoresearch-rsi-keepalive@${INSTANCE}.service" \
  "autoresearch-rsi-supervisor@${INSTANCE}.service" \
  "autoresearch-rsi-monitor@${INSTANCE}.service"
```

## Status and logs

```bash
INSTANCE=rsi-autonomous-llm-self-improvement
sudo systemctl status \
  "autoresearch-rsi-supervisor@${INSTANCE}.service" \
  "autoresearch-rsi-monitor@${INSTANCE}.service" \
  "autoresearch-rsi-keepalive@${INSTANCE}.service" \
  "autoresearch-rsi-dashboard@${INSTANCE}.service"
sudo journalctl -fu "autoresearch-rsi-supervisor@${INSTANCE}.service"
sudo journalctl -fu "autoresearch-rsi-monitor@${INSTANCE}.service"
sudo journalctl -fu "autoresearch-rsi-keepalive@${INSTANCE}.service"
sudo journalctl -fu "autoresearch-rsi-dashboard@${INSTANCE}.service"
```

The dashboard is intentionally bound to loopback. Open it locally at
`http://127.0.0.1:8099/` or reach it through the existing authenticated tunnel.

Application logs remain in the campaign and ClusterBridge state paths.
Wrapper stdout/stderr is available through `journalctl`; component-maintained
keepalive logs remain under `/var/log/autoresearch-rsi/<id>/`.

## Safe pause, resume, and permanent stop

`systemctl stop` on the supervisor only stops the managed process. It does not
write a permanent campaign stop marker, because that would make a transient
service restart or machine shutdown resume-hostile. For an operator pause,
request the durable pause first and then stop the service:

```bash
INSTANCE=rsi-autonomous-llm-self-improvement
/data/workspace/autoresearch-stack/AutoResearchClaw/bin/rsi-pause \
  /root/shared/.clusters/.workdir/autoresearch-rsi/$INSTANCE \
  "operator maintenance"
```

The supervisor exits cooperatively at the next control point. Resume with:

```bash
sudo systemctl start "autoresearch-rsi-supervisor@${INSTANCE}.service"
```

For a permanent campaign stop, use the application command, then stop the
monitor and keepalive services if the allocation should no longer be retained:

```bash
/data/workspace/autoresearch-stack/AutoResearchClaw/bin/rsi-stop \
  /root/shared/.clusters/.workdir/autoresearch-rsi/$INSTANCE \
  --reason "operator ended campaign"
sudo systemctl stop "autoresearch-rsi-monitor@${INSTANCE}.service"
sudo systemctl stop "autoresearch-rsi-keepalive@${INSTANCE}.service"
```

Stopping the keepalive only stops renewal; it deliberately does not invoke
`alloc-release`, `release`, node cleanup, Ray stop, or SSH.

## Static validation

```bash
systemd-analyze verify \
  deploy/systemd/autoresearch-rsi.target \
  deploy/systemd/autoresearch-rsi-alert@.service \
  deploy/systemd/autoresearch-rsi-supervisor@.service \
  deploy/systemd/autoresearch-rsi-monitor@.service \
  deploy/systemd/autoresearch-rsi-keepalive@.service
```
