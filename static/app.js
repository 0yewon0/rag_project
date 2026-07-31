const messagesEl = document.querySelector("#messages");
const formEl = document.querySelector("#chatForm");
const inputEl = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const resetButton = document.querySelector("#resetButton");
const statusBar = document.querySelector("#statusBar");

let sessionId = localStorage.getItem("financial_chat_session_id");

const productTypeLabels = {
  deposit: "정기예금",
  saving: "적금",
};

const ratePreferenceLabels = {
  base_rate: "기본금리",
  max_rate: "최고우대금리",
};

function addMessage(role, text) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  message.appendChild(bubble);
  messagesEl.appendChild(message);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  sendButton.textContent = isLoading ? "응답 중" : "전송";
  inputEl.disabled = isLoading;
}

function updateStatus(data = {}) {
  const productType = productTypeLabels[data.product_type] || "상품유형 미정";
  const term = data.term_months ? `${data.term_months}개월` : "기간 미정";
  const rate =
    ratePreferenceLabels[data.rate_preference] || "금리기준 미정";
  const monthlyAmount = data.monthly_amount
    ? `월 ${Number(data.monthly_amount).toLocaleString("ko-KR")}원`
    : "납입액 미정";
  const card = data.card_ok === false ? "카드 제외" : "카드 조건 미정";
  const salary =
    data.salary_transfer_ok === false ? "급여이체 제외" : "급여이체 미정";
  const autoTransfer =
    data.auto_transfer_ok === false ? "자동이체 제외" : "자동이체 미정";
  const mobile =
    data.mobile_join_preferred === true ? "모바일 선호" : "가입방식 미정";

  statusBar.innerHTML = "";
  for (const label of [
    productType,
    term,
    rate,
    monthlyAmount,
    card,
    salary,
    autoTransfer,
    mobile,
  ]) {
    const item = document.createElement("span");
    item.textContent = label;
    statusBar.appendChild(item);
  }
}

async function sendMessage(message) {
  setLoading(true);
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        session_id: sessionId,
        message,
      }),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || "요청에 실패했습니다.");
    }

    const data = await response.json();
    sessionId = data.session_id;
    localStorage.setItem("financial_chat_session_id", sessionId);

    addMessage("bot", data.answer);
    updateStatus(data);
  } catch (error) {
    addMessage("bot", `오류가 발생했습니다: ${error.message}`);
  } finally {
    setLoading(false);
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const message = inputEl.value.trim();
  if (!message) return;

  addMessage("user", message);
  inputEl.value = "";
  sendMessage(message);
});

resetButton.addEventListener("click", async () => {
  if (sessionId) {
    await fetch(`/reset?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
    }).catch(() => {});
  }
  sessionId = crypto.randomUUID();
  localStorage.setItem("financial_chat_session_id", sessionId);
  messagesEl.innerHTML = "";
  updateStatus();
  addMessage("bot", "안녕하세요. 원하시는 예금이나 적금 조건을 알려주세요.");
  inputEl.focus();
});

if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem("financial_chat_session_id", sessionId);
}

updateStatus();
addMessage("bot", "안녕하세요. 원하시는 예금이나 적금 조건을 알려주세요.");
inputEl.focus();
