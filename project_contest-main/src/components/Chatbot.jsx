import React, { useState } from "react";
import { MessageCircle, X, Send } from "lucide-react";

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { from: "bot", text: "👋 Hi! How can I help you today?" },
  ]);
  const [input, setInput] = useState("");

  const toggleChat = () => setIsOpen(!isOpen);

  const sendMessage = () => {
    if (input.trim() === "") return;
    setMessages((m) => [...m, { from: "user", text: input }]);
    setInput("");
    // NOTE: integrate your bot API here if desired
  };

  return (
    <div className="chatbot-container">
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="bg-primary text-white p-3 rounded-full shadow-lg"
          aria-label="Open chat"
        >
          <MessageCircle size={20} />
        </button>
      )}

      {isOpen && (
        <div className="chat-window w-80 h-96 bg-white shadow-2xl rounded-2xl flex flex-col overflow-hidden border border-gray-200">
          <div className="bg-primary text-white d-flex justify-content-between align-items-center px-3 py-2">
            <h6 className="mb-0">Chat with us!</h6>
            <button onClick={toggleChat} className="btn btn-sm btn-link text-white">
              <X size={16} />
            </button>
          </div>

          <div className="flex-grow-1 p-3 overflow-auto" style={{ background: "var(--bs-body-bg)" }}>
            {messages.map((msg, i) => (
              <div key={i} className={`d-flex mb-2 ${msg.from === "user" ? "justify-content-end" : ""}`}>
                <div
                  style={{ maxWidth: "75%" }}
                  className={`p-2 rounded-3 ${msg.from === "user" ? "bg-primary text-white" : "bg-light text-dark"}`}
                >
                  {msg.text}
                </div>
              </div>
            ))}
          </div>

          <div className="d-flex align-items-center border-top p-2 bg-white">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="form-control me-2"
            />
            <button onClick={sendMessage} className="btn btn-primary">
              <Send size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
