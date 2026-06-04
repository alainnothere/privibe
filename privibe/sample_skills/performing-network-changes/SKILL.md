---
name: performing-network-changes
description: "Analyze, Review, Perform, and Validate network changes. 
Use this skill for any network-related change such as modifying IP addresses, changing interfaces from DHCP to static, editing routing rules, 
or any other network configuration modification."
user-invocable: true
---

# Network Changes Skill

This skill MUST be followed step by step, in order, without skipping steps, whenever a network-related change is required. 
Do not proceed to the next step until the current step is complete.

Network-related changes include but are not limited to:
- Changing an IP address on any interface
- Converting an interface from DHCP to static
- Modifying routing rules or default gateways
- Changing DNS settings
- Modifying firewall rules that affect connectivity
- Any change to network configuration files

---

## Step 1 — Understand the change

- State in plain language what you understand the requested change to be.
- If anything is ambiguous, ask a clarifying question. Do not assume.
- Do not proceed until you are confident you understand the exact intended change and nothing more.

## Step 2 — Analyze scope and side effects

- Identify every file, interface, service, or system that could be affected by this change.
- Identify whether related changes are required (e.g., changing an IP may also require updating DNS, firewall rules, or dependent service configs).
- State all of this explicitly to the user before proceeding.

## Step 3 — Review the current state in two independent ways

Read the current configuration using at least two independent methods. Examples:
- Read the configuration file directly (e.g., `cat /etc/network/interfaces`)
- Read the live system state (e.g., `ip addr show`, `ip route`, `uci show network`)

Both reads must agree. If they disagree, stop and report the discrepancy to the user before proceeding.

## Step 4 — Define rollback

- State explicitly how the change can be undone if something goes wrong.
- If the change could cause loss of remote access (e.g., SSH over the interface being changed), flag this to the user and confirm they have an alternative access path or accept the risk.
- Do not proceed until rollback is defined or the user explicitly waives it.

## Step 5 — Define validation

- State explicitly how the change will be verified after it is applied (e.g., `ping`, `ip addr`, checking a service, confirming remote access still works).
- Do not proceed until validation criteria are defined or the user explicitly waives them.

## Step 6 — Prepare the change and show it to the user

- Show exactly what will be changed: the before state and the after state (a diff or equivalent).
- The change must be scoped so that ONLY the intended value is modified. If using a tool like `sed`, verify the pattern is specific enough that it cannot match unintended lines. Show the exact command or edit that will be applied.
- Do not apply the change yet.
- Ask the user to confirm before proceeding.

## Step 7 — Apply the change

- Use the `safe-file-edit` skill if the change involves modifying a configuration file.
- Apply only what was shown in Step 6. Nothing else.

## Step 8 — Validate

- Execute the validation steps defined in Step 5.
- Show the output to the user.
- If validation fails, stop and report. Do not attempt further changes without explicit instruction.

## Step 9 — Confirm and summarize

- Confirm the change is applied and validated.
- Summarize what was changed, what was verified, and note any follow-up items identified during analysis.




---
name: performing-network-changes
description: Analyze Review Perform and Validate network changes, see also the safe-file-edit skill
user-invocable: true
---

# Code Review Skill

This skill should be used whenever you are required to perform a network related change.
Network related changes could be:
- changing an ip address
- make an interface that used dhcp, to now use a static ip address
- etc..

When that happens YOU SHOULD OBSERVE:
- Change required make sense, don't assume, ask question if in doubt, push back unless confirmed you should go ahead
- Analyze if other related changes are required
- Review the existing configuration in more than 1 way to confirm the change to be applied
- Validate with the user how the change will be validated after being applied, and proceed only it has been defined, or the user request the change to be performed
- Ask the user to confirm the change before making it
- Change should be applied always in a way that it's ensure no more than the intended change is applied, for example if you use sed to change the ip address of an interface in an openwrt router you could change all the ip addresses in the router, because of that, you should first review the configuration file to be changed, and ENSURE THE CHANGE WILL BE DONE IN A WAY THAT DOESN'T CHANGE ANYTHING ELSE
- Validate the change
