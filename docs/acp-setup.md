# ACP Setup

Privibe can be used in text editors and IDEs that support [Agent Client Protocol](https://agentclientprotocol.com/overview/clients). Privibe includes the `privibe-acp` tool.
Once you have set up `privibe` with the API keys, you are ready to use `privibe-acp` in your editor. Below are the setup instructions for some editors that support ACP.

## Zed

For usage in Zed, you can set up a local install as follows:

1. Go to `~/.config/zed/settings.json` and, under the `agent_servers` JSON object, add the following key-value pair to invoke the `vibe-acp` command. Here is the snippet:

```json
{
   "agent_servers": {
        "Privibe": {
         "type": "custom",
         "command": "privibe-acp",
         "args": [],
         "env": {}
      }
   }
}
```

1. In the `New Thread` pane on the right, select the `Privibe` agent and start the conversation.

## JetBrains IDEs

For using Privibe in JetBrains IDEs, you'll need to have the [Jetbrains AI Assistant extension](https://plugins.jetbrains.com/plugin/22282-jetbrains-ai-assistant) installed

### Version 2025.3 or later

1. Open settings, then go to `Tools > AI Assistant > Agents`. Search for `Privibe`, click install

2. Open AI Assistant. You should be able to select Privibe from the agent selector (if you're not authenticated yet, you will be prompted to do so).

### Legacy method

1. Add the following snippet to your JetBrains IDE acp.json ([documentation](https://www.jetbrains.com/help/ai-assistant/acp.html)):

```json
{
  "agent_servers": {
      "Privibe": {
      "command": "privibe-acp",
    }
  }
}
```

1. In the AI Chat agent selector, select the new Privibe agent and start the conversation.

## Neovim (using avante.nvim)

Add Privibe in the acp_providers section of your configuration

```lua
{
  acp_providers = {
      ["privibe"] = {
        command = "privibe-acp",
      env = {
           LLM_API_KEY = os.getenv("LLM_API_KEY"), -- necessary if you setup Privibe manually
      },
    }
  }
}
```
