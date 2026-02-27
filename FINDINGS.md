# Go Fast Contracts: On-Chain Deployment Investigation

**Date:** 2026-02-27
**Context:** Continuing investigation from lost session on KJs-Mac-mini:kjm:9953

## The Vulnerability

**PR #8** ([skip-mev/go-fast-contracts#8](https://github.com/skip-mev/go-fast-contracts/pull/8)) — merged **2025-12-18** by `thal0x`

The `initiate_settlement` function in the CosmWasm `fast-transfer-gateway` contract did not validate the length of the `repayment_address` parameter. An attacker could submit a malformed address (e.g., 64 bytes instead of the required 32) during settlement, potentially causing funds to be sent to an unrecoverable or attacker-controlled address.

### Fix Applied (GitHub)

```rust
// cosmwasm/contracts/fast-transfer-gateway/src/execute.rs
pub fn initiate_settlement(...) -> ContractResponse {
    let config = CONFIG.load(deps.storage)?;

    // NEW: validate repayment address is exactly 32 bytes
    if repayment_address.len() != 32 {
        return Err(ContractError::InvalidRepaymentAddress);
    }
    // ...
}
```

A new error variant `ContractError::InvalidRepaymentAddress` was added, along with a test case `test_initiate_settlement_fails_if_repayment_address_is_invalid`.

## Deployed Contract Inventory

### CosmWasm (Osmosis, domain 875)

| Contract | Address | Purpose |
|----------|---------|---------|
| Fast Transfer Gateway | `osmo1vy34lpt5zlj797w7zqdta3qfq834kapx88qtgudy7jgljztj567s73ny82` | Main gateway — settlement, order filling |
| CW-7683 | `osmo1s8q6qwl7hz7xnfwsrrswzfrjk04q9ke8nr2kz5323ksdwjjue3vsxajqjg` | ERC-7683 compatible entry point |
| Previous Gateway | `osmo1cnze5c4y7jw69ghzczsnu9v9qz3xuvevw5ayr2g0pa3ayafumlusej3pf5` | Old address (referenced in deployCW7683.ts) |

**Token:** USDC via IBC (`ibc/498A0751C798A0D9A389AA3691123DADA57DAA4FE165D5C75894505B876BA6E4`)
**Mailbox (Hyperlane):** `osmo1jjf788v9m5pcqghe0ky2hf4llxxe37dqz6609eychuwe3xzzq9eql969h3`
**Hook:** `osmo13yswqchwtmv2ln9uz4w3865sfy5k8x0wg9qrv4vxflxjg0kuwwyqqpvqxz`

### Solidity (Arbitrum, domain 42161)

| Contract | Address | Purpose |
|----------|---------|---------|
| FastTransferGateway (EVM) | Proxied via ERC-1967 | Settlement counterpart |
| GoFastCaller | `0xF7ceC3d387384bB6cE5792dAb161a65cFaCf8aB4` | Referenced in setRemote cross-chain config |
| ISM | `0xb49a14568f9CC440f2c7DCf7FC6766040a5eb860` | Interchain Security Module |

**Owner:** `0x56Ca414d41CD3c1188A4939b0D56417dA7Bb6DA2`
**USDC (Arbitrum):** `0xaf88d065e77c8cC2239327C5EDb3A432268e5831`

## Critical Question: Was the Contract Actually Migrated On-Chain?

### Evidence the migration script was PREPARED

1. `migrate.ts` was updated in PR #8 to point to the current gateway address (`osmo1vy34lpt...`)
2. Gas price was bumped from `0.025uosmo` to `0.1uosmo` (suggests awareness of current gas market)
3. The script stores a new WASM binary and calls `migrateContract`

### Evidence the migration may NOT have been executed

1. **No on-chain query was possible from this environment** — Cosmos LCD endpoints are blocked by egress proxy
2. **No migration transaction hash** is recorded anywhere in the repository
3. **No CHANGELOG, release tag, or deployment receipt** exists in the repo
4. The PR was created and merged within **2 minutes** (19:33 → 19:35 UTC on Dec 18) — extremely fast for a security-critical change
5. The repo has had **zero commits since the Dec 18 merge** — over 2 months of inactivity

### What Needs Verification (from a machine with chain access)

Run this on your Mac mini or any machine with Cosmos LCD access:

```bash
# Check Osmosis gateway contract history (code migrations)
curl -s "https://lcd.osmosis.zone/cosmwasm/wasm/v1/contract/osmo1vy34lpt5zlj797w7zqdta3qfq834kapx88qtgudy7jgljztj567s73ny82/history" | jq .

# Check current code_id
curl -s "https://lcd.osmosis.zone/cosmwasm/wasm/v1/contract/osmo1vy34lpt5zlj797w7zqdta3qfq834kapx88qtgudy7jgljztj567s73ny82" | jq '.contract_info.code_id'

# Check admin (who can migrate)
curl -s "https://lcd.osmosis.zone/cosmwasm/wasm/v1/contract/osmo1vy34lpt5zlj797w7zqdta3qfq834kapx88qtgudy7jgljztj567s73ny82" | jq '.contract_info.admin'

# If you find the code_id, download and compare the WASM:
# curl -s "https://lcd.osmosis.zone/cosmwasm/wasm/v1/code/{CODE_ID}" | jq -r '.data' | base64 -d | sha256sum
```

If the contract has only been instantiated once (single entry in history) and the code_id matches the original deployment, the fix has **not** been migrated on-chain.

## Risk Assessment

| Scenario | Impact |
|----------|--------|
| Migration executed | Low risk — vulnerability patched |
| Migration NOT executed | **High risk** — solvers can submit malformed repayment addresses during settlement, potentially stealing funds or griefing the protocol |
| EVM side | The Solidity contract is behind an ERC-1967 proxy (upgradeable by owner `0x56Ca...`), but PR #8 only modified CosmWasm code |

## Commit History (Full Timeline)

```
2025-12-18  65f061c  Merge PR #8: validate repayment address length
2025-12-18  d39ef04  validate repayment address length
2025-12-18  1c76fd9  validate repayment address length
2024-12-17  7a05185  Merge PR #7: enforce timeout is future
2024-12-17  cca68b1  enforce timeout timestamp is in the future
2024-12-04  8421c15  Merge PR #6: deploy instructions
2024-11-06  5c41bd4  Merge PR #5: address audit findings
2024-11-06  f12611f  only allow gateway as caller in GoFastCaller
2024-10-31  f9ccbef  forge install: skip-go-evm-contracts
...
2024-10-13  f0f49ad  setup new repo
```

42 total commits. Last activity: December 18, 2025.
