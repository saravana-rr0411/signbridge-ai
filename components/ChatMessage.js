// Chat Message Bubble Component for Admin and Deaf Person

export function renderChatMessage(msg) {
  const isAdmin = msg.sender === 'admin';

  if (isAdmin) {
    const activeStyle = msg.isActiveReply
      ? 'bg-secondary/10 border-secondary/30 ring-1 ring-secondary/20'
      : 'bg-surface-container-low border-outline-variant/50';

    return `
      <div class="flex flex-col gap-1 items-start w-full animate-fadeIn">
        <div class="flex items-center gap-1.5 text-xs text-on-surface-variant pl-1">
          <span class="material-symbols-outlined text-[15px] text-secondary">support_agent</span>
          <span class="font-bold text-primary">${msg.senderName || 'Admin (Officer Vance)'}</span>
          <span class="text-[10px] text-outline font-mono">${msg.time || 'Just now'}</span>
        </div>
        <div class="max-w-[90%] ${activeStyle} border text-on-surface rounded-2xl rounded-tl-xs px-3.5 py-2.5 shadow-xs">
          <p class="text-sm font-medium leading-relaxed break-words">${escapeHtml(msg.text)}</p>
        </div>
      </div>
    `;
  } else {
    return `
      <div class="flex flex-col gap-1 items-end w-full animate-fadeIn">
        <div class="flex items-center gap-1.5 text-xs text-on-surface-variant pr-1">
          <span class="text-[10px] text-outline font-mono">${msg.time || 'Just now'}</span>
          <span class="font-bold text-primary">${msg.senderName || 'Deaf Person'}</span>
          <span class="material-symbols-outlined text-[15px] text-primary">visibility</span>
        </div>
        <div class="max-w-[90%] bg-white border-2 border-primary/20 text-primary rounded-2xl rounded-tr-xs px-3.5 py-2.5 shadow-xs">
          <p class="text-sm font-semibold leading-relaxed break-words">"${escapeHtml(msg.text)}"</p>
        </div>
      </div>
    `;
  }
}

function escapeHtml(string) {
  return String(string)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
