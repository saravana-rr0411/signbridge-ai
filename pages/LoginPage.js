// Functional Login Page View matching Stitch Design
import { ParticleBackground } from '../components/ParticleBackground.js';
import { sessionStore } from '../state/sessionStore.js';

export function renderLoginPage() {
  return `
    <div class="min-h-screen w-full relative bg-[#f8fafc] text-[#191c1e] overflow-hidden flex items-center justify-center">
      <!-- Full-Screen Continuous Roaming Particle Field Canvas -->
      <canvas id="particleCanvas" class="fixed inset-0 pointer-events-none z-0 w-full h-full"></canvas>

      <!-- Main Split Layout (Exactly 50% / 50% Desktop) -->
      <main class="min-h-screen w-full grid grid-cols-1 md:grid-cols-2 relative z-10">
        <!-- Left Half: Project Branding (Cropped Pure Hand Symbol Arch + Title ONLY) -->
        <div class="flex flex-col items-center justify-center p-8 md:p-14 lg:p-20">
          <div class="flex flex-col items-center justify-center text-center">
            <!-- Clean isolated hand symbol: embedded text cropped out completely, pure transparent blend without white box -->
            <div class="w-[220px] h-[155px] overflow-hidden flex items-start justify-center" style="mix-blend-mode: multiply;">
              <img src="./assets/images/signbridge-logo.png" 
                   alt="SignBridge AI Hand Symbol" 
                   class="w-[220px] h-auto object-cover pointer-events-none select-none" 
                   style="clip-path: inset(0px 0px 38% 0px); mix-blend-mode: multiply; filter: contrast(1.12);" />
            </div>
            <!-- Separate Crisp HTML Title -->
            <h1 class="text-4xl lg:text-5xl font-extrabold tracking-tight text-[#0f2942] mt-4">
              SignBridge AI
            </h1>
          </div>
        </div>

        <!-- Right Half: Clean Login Form -->
        <div class="flex items-center justify-center p-6 sm:p-8 md:p-12 lg:p-16">
          <div class="max-w-md w-full px-7 sm:px-9 py-10 rounded-2xl bg-white/95 backdrop-blur-md border border-slate-200/90 shadow-[0_4px_24px_rgba(15,41,66,0.06)]">
            <!-- Header row with hand sign badge and Login heading -->
            <div class="flex items-center gap-3 mb-7">
              <div class="w-10 h-10 rounded-xl bg-teal-50 border border-teal-100 flex items-center justify-center text-[#006a61] shadow-xs flex-shrink-0">
                <svg aria-hidden="true" class="w-5 h-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
                  <path d="M18 11V6a2 2 0 0 0-4 0v4"></path>
                  <path d="M14 10V4a2 2 0 0 0-4 0v7"></path>
                  <path d="M10 10.5V6a2 2 0 0 0-4 0v8"></path>
                  <path d="M18 8a2 2 0 0 1 4 4v4a8 8 0 0 1-16 0v-2"></path>
                </svg>
              </div>
              <h2 class="text-2xl sm:text-3xl font-bold text-[#0f2942]">
                Login
              </h2>
            </div>

            <!-- Login Form -->
            <form id="login-form" class="space-y-5" novalidate>
              <!-- 1. Email Address -->
              <div>
                <label for="email" class="block text-sm font-semibold text-[#0f2942] mb-1.5">
                  Email Address
                </label>
                <input type="email" 
                       id="email" 
                       name="email" 
                       required 
                       placeholder="Enter your email" 
                       value="staff.vance@civicdesk.org"
                       class="w-full py-3 px-4 text-base border border-slate-300 rounded-lg focus:border-[#0f2942] focus:ring-1 focus:ring-[#0f2942] focus:outline-none text-slate-900 placeholder:text-slate-400 bg-white transition-colors" />
                <p id="email-error" class="hidden text-xs text-red-600 mt-1 font-medium"></p>
              </div>

              <!-- 2. Mobile Number -->
              <div>
                <label for="mobile" class="block text-sm font-semibold text-[#0f2942] mb-1.5">
                  Mobile Number
                </label>
                <div class="relative">
                  <span aria-hidden="true" class="absolute inset-y-0 left-0 pl-3.5 flex items-center pointer-events-none text-slate-400">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewBox="0 0 24 24">
                      <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path>
                    </svg>
                  </span>
                  <input type="tel" 
                         id="mobile" 
                         name="mobile" 
                         required 
                         placeholder="Enter your mobile number" 
                         value="+1 (555) 234-8901"
                         class="w-full py-3 pl-10 pr-4 text-base border border-slate-300 rounded-lg focus:border-[#0f2942] focus:ring-1 focus:ring-[#0f2942] focus:outline-none text-slate-900 placeholder:text-slate-400 bg-white transition-colors" />
                </div>
                <p id="mobile-error" class="hidden text-xs text-red-600 mt-1 font-medium"></p>
              </div>

              <!-- 3. Password -->
              <div>
                <label for="password" class="block text-sm font-semibold text-[#0f2942] mb-1.5">
                  Password
                </label>
                <div class="relative">
                  <input type="password" 
                         id="password" 
                         name="password" 
                         required 
                         placeholder="Enter your password" 
                         value="CivicAccess2026!"
                         class="w-full py-3 pl-4 pr-12 text-base border border-slate-300 rounded-lg focus:border-[#0f2942] focus:ring-1 focus:ring-[#0f2942] focus:outline-none text-slate-900 placeholder:text-slate-400 bg-white transition-colors" />
                  <button type="button" 
                          id="toggle-password" 
                          aria-label="Show password" 
                          class="absolute right-0 top-0 bottom-0 px-3.5 flex items-center justify-center text-slate-400 hover:text-[#0f2942] focus:outline-none cursor-pointer">
                    <!-- Eye Icon (Hidden state) -->
                    <svg id="eye-icon" aria-hidden="true" class="w-5 h-5" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" viewBox="0 0 24 24">
                      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path>
                      <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                    <!-- Eye Off Icon (Shown state) -->
                    <svg id="eye-off-icon" aria-hidden="true" class="w-5 h-5 hidden" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" viewBox="0 0 24 24">
                      <path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"></path>
                      <path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"></path>
                      <path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"></path>
                      <line x1="2" x2="22" y1="2" y2="22"></line>
                    </svg>
                  </button>
                </div>
                <p id="password-error" class="hidden text-xs text-red-600 mt-1 font-medium"></p>
              </div>

              <!-- Submit Button -->
              <div class="pt-2">
                <button type="submit" 
                        id="login-btn"
                        class="w-full h-12 bg-[#0f2942] hover:bg-[#1a3d5e] active:bg-[#091a2b] text-white rounded-lg font-semibold text-base transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-[#0f2942] focus:ring-offset-2 cursor-pointer shadow-sm">
                  Login
                </button>
              </div>
            </form>
          </div>
        </div>
      </main>
    </div>
  `;
}

export function initLoginPage() {
  // Init Particle Canvas
  const canvas = document.getElementById('particleCanvas');
  let particleBg = null;
  if (canvas) {
    particleBg = new ParticleBackground(canvas);
  }

  // Password Visibility Toggle
  const passwordInput = document.getElementById('password');
  const toggleBtn = document.getElementById('toggle-password');
  const eyeIcon = document.getElementById('eye-icon');
  const eyeOffIcon = document.getElementById('eye-off-icon');

  if (toggleBtn && passwordInput) {
    toggleBtn.addEventListener('click', () => {
      const isPassword = passwordInput.getAttribute('type') === 'password';
      if (isPassword) {
        passwordInput.setAttribute('type', 'text');
        toggleBtn.setAttribute('aria-label', 'Hide password');
        eyeIcon.classList.add('hidden');
        eyeOffIcon.classList.remove('hidden');
      } else {
        passwordInput.setAttribute('type', 'password');
        toggleBtn.setAttribute('aria-label', 'Show password');
        eyeIcon.classList.remove('hidden');
        eyeOffIcon.classList.add('hidden');
      }
    });
  }

  // Form Fields & Inline Error Elements
  const form = document.getElementById('login-form');
  const emailInput = document.getElementById('email');
  const mobileInput = document.getElementById('mobile');
  const emailError = document.getElementById('email-error');
  const mobileError = document.getElementById('mobile-error');
  const passwordError = document.getElementById('password-error');

  function clearErrors() {
    [emailError, mobileError, passwordError].forEach(el => {
      if (el) {
        el.textContent = '';
        el.classList.add('hidden');
      }
    });
    [emailInput, mobileInput, passwordInput].forEach(el => {
      if (el) {
        el.classList.remove('border-red-500', 'focus:border-red-500', 'focus:ring-red-500');
      }
    });
  }

  function showError(inputEl, errorEl, message) {
    if (errorEl) {
      errorEl.textContent = message;
      errorEl.classList.remove('hidden');
    }
    if (inputEl) {
      inputEl.classList.add('border-red-500', 'focus:border-red-500', 'focus:ring-red-500');
    }
  }

  // Handle Form Submission with Validation
  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      clearErrors();

      const email = emailInput ? emailInput.value.trim() : '';
      const mobile = mobileInput ? mobileInput.value.trim() : '';
      const password = passwordInput ? passwordInput.value.trim() : '';

      let hasError = false;

      if (!email) {
        showError(emailInput, emailError, 'Email Address cannot be empty');
        hasError = true;
      } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        showError(emailInput, emailError, 'Please enter a valid email address');
        hasError = true;
      }

      if (!mobile) {
        showError(mobileInput, mobileError, 'Mobile Number cannot be empty');
        hasError = true;
      }

      if (!password) {
        showError(passwordInput, passwordError, 'Password cannot be empty');
        hasError = true;
      }

      if (hasError) return;

      // Authenticate via sessionStore
      const result = sessionStore.login({ email, mobile, password });
      if (result.success) {
        // Navigate to Deaf Person Interface
        window.location.hash = '#/deaf';
      } else {
        showError(passwordInput, passwordError, result.error || 'Authentication failed');
      }
    });
  }

  return () => {
    if (particleBg) particleBg.destroy();
  };
}
