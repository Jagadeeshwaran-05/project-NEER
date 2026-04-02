import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Box,
  Chip,
  Fab,
  IconButton,
  Paper,
  Stack,
  TextField,
  Typography,
  Divider,
  CircularProgress,
} from "@mui/material";
import {
  Chat as ChatIcon,
  Close as CloseIcon,
  Send as SendIcon,
} from "@mui/icons-material";
import {
  ChatMessage,
  Lake,
  sendChatMessage,
} from "../services/apiService";

interface ChatAssistantProps {
  lakes: Lake[];
  selectedLake: Lake | null;
  selectedYear: number;
}

const quickPrompts = [
  "Why is this lake in poor condition?",
  "What should we do next?",
  "Summarize the risk in one line.",
  "Compare this year with last year.",
];

const ChatAssistant: React.FC<ChatAssistantProps> = ({ lakes, selectedLake, selectedYear }) => {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hi! I can explain the lake data, trends, and actions. Ask me about the selected lake or the dashboard metrics.",
    },
  ]);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const activeLake = useMemo(() => {
    return selectedLake || lakes[0] || null;
  }, [selectedLake, lakes]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  const sendMessage = async (messageText?: string) => {
    const finalMessage = (messageText || input).trim();
    if (!finalMessage || loading) {
      return;
    }

    const nextMessages: ChatMessage[] = [
      ...messages,
      { role: "user", content: finalMessage },
    ];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await sendChatMessage({
        message: finalMessage,
        history: messages,
        year: selectedYear,
        lake: activeLake
          ? {
              id: activeLake.id,
              name: activeLake.name,
              year: activeLake.year,
              waterHealth: activeLake.waterHealth,
              ndwi: activeLake.ndwi,
              ndci: activeLake.ndci,
              fai: activeLake.fai,
              mci: activeLake.mci,
              swir_ratio: activeLake.swir_ratio,
              turbidity: activeLake.turbidity,
              bodLevel: activeLake.bodLevel,
              pollutionCauses: activeLake.pollutionCauses,
              suggestions: activeLake.suggestions,
            }
          : null,
      });

      setMessages((current) => [
        ...current,
        { role: "assistant", content: response.reply },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: "I couldn't reach the local assistant right now. Please try again after starting Ollama.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {!open && (
        <Fab
          color="primary"
          onClick={() => setOpen(true)}
          sx={{
            position: "fixed",
            right: 24,
            bottom: 24,
            zIndex: 1400,
          }}
        >
          <ChatIcon />
        </Fab>
      )}

      {open && (
        <Paper
          elevation={8}
          sx={{
            position: "fixed",
            right: 24,
            bottom: 24,
            width: 360,
            maxWidth: "calc(100vw - 32px)",
            height: 520,
            display: "flex",
            flexDirection: "column",
            zIndex: 1400,
            overflow: "hidden",
            borderRadius: 3,
          }}
        >
          <Box
            sx={{
              px: 2,
              py: 1.5,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              bgcolor: "primary.main",
              color: "white",
            }}
          >
            <Box>
              <Typography variant="subtitle1" fontWeight={700}>
                Lake Assistant
              </Typography>
              <Typography variant="caption" sx={{ opacity: 0.9 }}>
                Local Ollama chat
              </Typography>
            </Box>
            <IconButton onClick={() => setOpen(false)} size="small" sx={{ color: "white" }}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Box>

          <Box sx={{ px: 2, py: 1.5 }}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              <Chip
                label={activeLake ? `${activeLake.name} • ${selectedYear}` : `All lakes • ${selectedYear}`}
                size="small"
                color="primary"
                variant="outlined"
              />
              {activeLake && (
                <Chip
                  label={activeLake.waterHealth}
                  size="small"
                  variant="outlined"
                />
              )}
            </Stack>
          </Box>

          <Divider />

          <Box
            sx={{
              flex: 1,
              p: 2,
              overflowY: "auto",
              bgcolor: "grey.50",
            }}
          >
            <Stack spacing={1.5}>
              {messages.map((message, index) => (
                <Box
                  key={`${message.role}-${index}`}
                  sx={{
                    alignSelf: message.role === "user" ? "flex-end" : "flex-start",
                    maxWidth: "85%",
                    bgcolor: message.role === "user" ? "primary.main" : "white",
                    color: message.role === "user" ? "white" : "text.primary",
                    px: 1.5,
                    py: 1,
                    borderRadius: 2,
                    boxShadow: 1,
                    whiteSpace: "pre-wrap",
                  }}
                >
                  <Typography variant="body2">{message.content}</Typography>
                </Box>
              ))}
              {loading && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, color: "text.secondary" }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2">Thinking...</Typography>
                </Box>
              )}
              <div ref={messagesEndRef} />
            </Stack>
          </Box>

          <Box sx={{ px: 2, pb: 1 }}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mb: 1 }}>
              {quickPrompts.map((prompt) => (
                <Chip
                  key={prompt}
                  label={prompt}
                  size="small"
                  onClick={() => sendMessage(prompt)}
                  sx={{ cursor: "pointer" }}
                />
              ))}
            </Stack>
            <Stack direction="row" spacing={1} alignItems="flex-end">
              <TextField
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                  }
                }}
                placeholder="Ask about lake health, trends, or actions..."
                fullWidth
                size="small"
                multiline
                minRows={1}
                maxRows={4}
              />
              <Fab
                color="primary"
                size="small"
                disabled={loading || !input.trim()}
                onClick={() => sendMessage()}
              >
                <SendIcon fontSize="small" />
              </Fab>
            </Stack>
          </Box>
        </Paper>
      )}
    </>
  );
};

export default ChatAssistant;
