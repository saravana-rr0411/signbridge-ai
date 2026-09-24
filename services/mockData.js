// Realistic Mock Data for SignBridge AI

export const PRESET_CATEGORIES = {
  HOSPITAL: 'HOSPITAL',
  BANK: 'BANK',
  GOV: 'GOVERNMENT OFFICE'
};

export const PRESETS = {
  [PRESET_CATEGORIES.HOSPITAL]: [
    { id: 'h1', text: 'Please wait here. The doctor will examine you shortly.', icon: 'hourglass_top', num: '#01' },
    { id: 'h2', text: 'Please take a seat in the waiting area.', icon: 'chair', num: '#02' },
    { id: 'h3', text: 'The doctor is currently with another patient and is coming.', icon: 'medical_services', num: '#03' },
    { id: 'h4', text: 'Please show your hospital ID or appointment ticket.', icon: 'badge', num: '#04' },
    { id: 'h5', text: 'What is your problem? Please describe your symptoms.', icon: 'help_outline', num: '#05' },
    { id: 'h6', text: 'Where is the pain located? Please point to it.', icon: 'vital_signs', num: '#06' },
    { id: 'h7', text: 'Please show your medical report and previous prescriptions.', icon: 'clinical_notes', num: '#07' },
    { id: 'h8', text: 'Please wait for your token number to be displayed.', icon: 'qr_code_2', num: '#08' },
    { id: 'h9', text: 'Please come with me to the examination room.', icon: 'directions_walk', num: '#09' },
    { id: 'h10', text: 'Emergency assistance has been requested immediately.', icon: 'e911_emergency', num: '#10', isPriority: true },
    { id: 'h11', text: 'Your turn is next. Please prepare your card.', icon: 'notification_important', num: '#11' },
    { id: 'h12', text: 'Please take this prescription to the pharmacy counter.', icon: 'medication', num: '#12' },
    { id: 'h13', text: 'Please fill out this patient admission form.', icon: 'description', num: '#13' },
    { id: 'h14', text: 'Please sign here to consent to examination.', icon: 'draw', num: '#14' },
    { id: 'h15', text: 'Thank you. Your consultation has been completed.', icon: 'sentiment_satisfied', num: '#15' }
  ],
  [PRESET_CATEGORIES.BANK]: [
    { id: 'b1', text: 'Please enter your 4-digit PIN on the terminal keypad.', icon: 'pin', num: '#01' },
    { id: 'b2', text: 'Please provide your bank passbook or statement.', icon: 'menu_book', num: '#02' },
    { id: 'b3', text: 'Cash withdrawal is ready for collection.', icon: 'payments', num: '#03' },
    { id: 'b4', text: 'Please take your queue token and wait for Counter 3.', icon: 'confirmation_number', num: '#04' },
    { id: 'b5', text: 'Your account KYC verification is complete.', icon: 'verified_user', num: '#05' },
    { id: 'b6', text: 'Please show an official government photo identity card.', icon: 'badge', num: '#06' },
    { id: 'b7', text: 'Please sign on the designated line on this withdrawal voucher.', icon: 'draw', num: '#07' },
    { id: 'b8', text: 'Would you like a printed deposit receipt for this transaction?', icon: 'receipt', num: '#08' },
    { id: 'b9', text: 'Your new ATM debit card has been dispatched to your address.', icon: 'credit_card', num: '#09' },
    { id: 'b10', text: 'Please verify your registered mobile number on screen.', icon: 'phone_android', num: '#10' },
    { id: 'b11', text: 'Your cheque has been submitted for clearing.', icon: 'fact_check', num: '#11' },
    { id: 'b12', text: 'Please fill out this cash deposit slip.', icon: 'description', num: '#12' },
    { id: 'b13', text: 'Your account balance summary has been printed.', icon: 'print', num: '#13' },
    { id: 'b14', text: 'Foreign exchange service is available at Desk 6.', icon: 'currency_exchange', num: '#14' },
    { id: 'b15', text: 'Thank you for banking with us. Have a safe day.', icon: 'sentiment_satisfied', num: '#15' }
  ],
  [PRESET_CATEGORIES.GOV]: [
    { id: 'g1', text: 'Your application is currently under official review.', icon: 'pending_actions', num: '#01' },
    { id: 'g2', text: 'Please proceed to Counter 07 for biometric verification.', icon: 'forward', num: '#02' },
    { id: 'g3', text: 'Your civic certificate has been officially approved.', icon: 'verified', num: '#03' },
    { id: 'g4', text: 'Please bring the original utility bill for address proof.', icon: 'receipt_long', num: '#04' },
    { id: 'g5', text: 'Official municipal stamp and seal affixed successfully.', icon: 'approval', num: '#05' },
    { id: 'g6', text: 'Please provide two recent passport-size photos.', icon: 'portrait', num: '#06' },
    { id: 'g7', text: 'Please sign this declaration form in front of the officer.', icon: 'draw', num: '#07' },
    { id: 'g8', text: 'Residency and citizenship verification has passed.', icon: 'how_to_reg', num: '#08' },
    { id: 'g9', text: 'Please pay the municipal service fee at Cashier Desk.', icon: 'point_of_sale', num: '#09' },
    { id: 'g10', text: 'A biometric fingerprint scan is required to proceed.', icon: 'fingerprint', num: '#10' },
    { id: 'g11', text: 'Your tracking reference number is active in civic database.', icon: 'pin_invoke', num: '#11' },
    { id: 'g12', text: 'Please present your original birth certificate or title deed.', icon: 'history_edu', num: '#12' },
    { id: 'g13', text: 'Standard renewal processing takes 3 to 5 business days.', icon: 'schedule', num: '#13' },
    { id: 'g14', text: 'Please wait in Zone B until your queue number is called.', icon: 'chair', num: '#14' },
    { id: 'g15', text: 'Thank you. Your civic service request has been processed.', icon: 'sentiment_satisfied', num: '#15' }
  ]
};

export const INITIAL_CONVERSATION = [
  {
    id: 'm1',
    sender: 'admin',
    senderName: 'Admin (Officer Vance)',
    text: 'Welcome to Desk #4. How can I assist you today?',
    time: '10:41 AM',
    timestamp: 1727154660000
  },
  {
    id: 'm2',
    sender: 'deaf',
    senderName: 'Deaf Person',
    text: 'I need help with my appointment.',
    time: '10:42 AM',
    timestamp: 1727154720000
  },
  {
    id: 'm3',
    sender: 'admin',
    senderName: 'Admin (Officer Vance)',
    text: 'Please show your appointment ID or ticket number.',
    time: '10:42 AM',
    timestamp: 1727154740000
  },
  {
    id: 'm4',
    sender: 'deaf',
    senderName: 'Deaf Person',
    text: 'Here is my ticket: #A-204.',
    time: '10:43 AM',
    timestamp: 1727154780000
  },
  {
    id: 'm5',
    sender: 'admin',
    senderName: 'Admin (Officer Vance)',
    text: 'Please wait here. The doctor will examine you shortly.',
    time: '10:43 AM',
    timestamp: 1727154800000,
    isActiveReply: true
  }
];

export const CHAT_HISTORY_RECORDS = [
  {
    id: 'SB-20260924-003',
    date: '2026-09-24',
    displayDate: '24 Sep 2026',
    time: '4:32 PM',
    status: 'Completed',
    counter: 'Desk #04',
    adminName: 'Officer J. Vance',
    adminPreview: 'Please wait here. The doctor will examine you shortly.',
    deafPreview: 'Okay, thank you.',
    messages: [
      { sender: 'admin', text: 'Welcome to Desk #4. How can I assist you today?', time: '4:28 PM' },
      { sender: 'deaf', text: 'I need help with my appointment.', time: '4:29 PM' },
      { sender: 'admin', text: 'Please show your appointment ID or ticket number.', time: '4:30 PM' },
      { sender: 'deaf', text: 'Here is my ticket: #A-204.', time: '4:31 PM' },
      { sender: 'admin', text: 'Please wait here. The doctor will examine you shortly.', time: '4:32 PM' },
      { sender: 'deaf', text: 'Okay, thank you.', time: '4:32 PM' }
    ]
  },
  {
    id: 'SB-20260924-002',
    date: '2026-09-24',
    displayDate: '24 Sep 2026',
    time: '1:15 PM',
    status: 'Completed',
    counter: 'Desk #04',
    adminName: 'Officer J. Vance',
    adminPreview: 'Please show your ID or ticket number.',
    deafPreview: 'Here is my ticket: #A-204.',
    messages: [
      { sender: 'admin', text: 'Hello, what civic department are you visiting?', time: '1:10 PM' },
      { sender: 'deaf', text: 'Health registration desk.', time: '1:12 PM' },
      { sender: 'admin', text: 'Please show your ID or ticket number.', time: '1:13 PM' },
      { sender: 'deaf', text: 'Here is my ticket: #A-204.', time: '1:15 PM' },
      { sender: 'admin', text: 'Thank you, please proceed to Room 12.', time: '1:15 PM' }
    ]
  },
  {
    id: 'SB-20260924-001',
    date: '2026-09-24',
    displayDate: '24 Sep 2026',
    time: '10:42 AM',
    status: 'Completed',
    counter: 'Desk #04',
    adminName: 'Officer J. Vance',
    adminPreview: 'Good morning. Are you applying for a residential permit or renewal?',
    deafPreview: 'Permit renewal for 2 years.',
    messages: [
      { sender: 'admin', text: 'Good morning. Are you applying for a residential permit or renewal?', time: '10:38 AM' },
      { sender: 'deaf', text: 'Permit renewal for 2 years.', time: '10:40 AM' },
      { sender: 'admin', text: 'Please submit your proof of address.', time: '10:41 AM' },
      { sender: 'deaf', text: 'Here is my electricity bill and copy.', time: '10:42 AM' }
    ]
  },
  {
    id: 'SB-20260923-002',
    date: '2026-09-23',
    displayDate: '23 Sep 2026',
    time: '3:20 PM',
    status: 'Completed',
    counter: 'Desk #02',
    adminName: 'Officer M. Ross',
    adminPreview: 'Your address verification document has been approved.',
    deafPreview: 'Thank you. Where do I collect the stamp?',
    messages: [
      { sender: 'admin', text: 'Your address verification document has been approved.', time: '3:18 PM' },
      { sender: 'deaf', text: 'Thank you. Where do I collect the stamp?', time: '3:20 PM' },
      { sender: 'admin', text: 'Counter 7 right down the hallway.', time: '3:20 PM' }
    ]
  },
  {
    id: 'SB-20260923-001',
    date: '2026-09-23',
    displayDate: '23 Sep 2026',
    time: '11:05 AM',
    status: 'Completed',
    counter: 'Desk #02',
    adminName: 'Officer M. Ross',
    adminPreview: 'Please confirm your contact phone number on the key display.',
    deafPreview: 'Confirmed, it is correct.',
    messages: [
      { sender: 'admin', text: 'Please confirm your contact phone number on the key display.', time: '11:04 AM' },
      { sender: 'deaf', text: 'Confirmed, it is correct.', time: '11:05 AM' }
    ]
  },
  {
    id: 'SB-20260922-001',
    date: '2026-09-22',
    displayDate: '22 Sep 2026',
    time: '2:10 PM',
    status: 'Completed',
    counter: 'Desk #04',
    adminName: 'Officer J. Vance',
    adminPreview: 'The queue fee is 15 dollars. Cash or card?',
    deafPreview: 'Card please.',
    messages: [
      { sender: 'admin', text: 'The queue fee is 15 dollars. Cash or card?', time: '2:08 PM' },
      { sender: 'deaf', text: 'Card please.', time: '2:09 PM' },
      { sender: 'admin', text: 'Please tap your card on the terminal.', time: '2:10 PM' }
    ]
  },
  {
    id: 'SB-20260918-001',
    date: '2026-09-18',
    displayDate: '18 Sep 2026',
    time: '11:45 AM',
    status: 'Completed',
    counter: 'Desk #04',
    adminName: 'Officer J. Vance',
    adminPreview: 'Your municipal voter registration card is ready for signature.',
    deafPreview: 'I have signed the confirmation form.',
    messages: [
      { sender: 'admin', text: 'Your municipal voter registration card is ready for signature.', time: '11:42 AM' },
      { sender: 'deaf', text: 'I have signed the confirmation form.', time: '11:45 AM' }
    ]
  },
  {
    id: 'SB-20260910-001',
    date: '2026-09-10',
    displayDate: '10 Sep 2026',
    time: '9:30 AM',
    status: 'Completed',
    counter: 'Desk #01',
    adminName: 'Officer S. Chen',
    adminPreview: 'Welcome to Social Services. Do you need an ASL certified interpreter?',
    deafPreview: 'No, SignBridge AI terminal is working well.',
    messages: [
      { sender: 'admin', text: 'Welcome to Social Services. Do you need an ASL certified interpreter?', time: '9:28 AM' },
      { sender: 'deaf', text: 'No, SignBridge AI terminal is working well.', time: '9:30 AM' }
    ]
  }
];
