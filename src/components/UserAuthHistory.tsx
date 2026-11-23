import { History, KeyRound, LogOut, Trash2, UserCheck } from 'lucide-react';
import { useEffect, useState } from 'react';

export interface SearchHistoryEntry {
  id: string;
  gene1: string;
  gene2: string;
  score?: number;
  rank?: number;
  searchedAt: string;
}

interface UserRecord {
  email: string;
  passkeyHash: string;
  credentialId?: string;
}

interface UserAuthHistoryProps {
  currentUserEmail: string | null;
  onUserChange: (email: string | null) => void;
  historyByUser: Record<string, SearchHistoryEntry[]>;
  onSelectHistoryItem: (g1: string, g2: string) => void;
  onClearHistory: (email: string) => void;
}

const USERS_STORAGE_KEY = 'bspai_users_v1';

async function hashPasskey(email: string, passkey: string): Promise<string> {
  const payload = new TextEncoder().encode(`${email.trim().toLowerCase()}::${passkey}`);
  const digest = await crypto.subtle.digest('SHA-256', payload);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
}

function loadUsers(): Record<string, UserRecord> {
  try {
    const raw = localStorage.getItem(USERS_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Record<string, UserRecord>) : {};
  } catch {
    return {};
  }
}

function saveUsers(users: Record<string, UserRecord>) {
  localStorage.setItem(USERS_STORAGE_KEY, JSON.stringify(users));
}

async function tryRegisterWebAuthnPasskey(email: string): Promise<string | undefined> {
  if (
    typeof window === 'undefined' ||
    !window.isSecureContext ||
    !window.PublicKeyCredential ||
    !navigator.credentials?.create
  ) {
    return undefined;
  }
  try {
    const userId = new TextEncoder().encode(email.trim().toLowerCase());
    const challenge = crypto.getRandomValues(new Uint8Array(32));
    const cred = (await navigator.credentials.create({
      publicKey: {
        challenge,
        rp: { name: 'Ubhaya-Bandha (BSPAI)' },
        user: {
          id: userId,
          name: email.trim().toLowerCase(),
          displayName: email.trim().toLowerCase(),
        },
        pubKeyCredParams: [{ type: 'public-key', alg: -7 }],
        authenticatorSelection: {
          authenticatorAttachment: 'platform',
          userVerification: 'preferred',
        },
        timeout: 20000,
      },
    })) as PublicKeyCredential | null;
    return cred?.id;
  } catch {
    return undefined;
  }
}

export function UserAuthHistory({
  currentUserEmail,
  onUserChange,
  historyByUser,
  onSelectHistoryItem,
  onClearHistory,
}: UserAuthHistoryProps) {
  const [emailInput, setEmailInput] = useState(currentUserEmail ?? '');
  const [passkeyInput, setPasskeyInput] = useState('');
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (currentUserEmail) {
      setEmailInput(currentUserEmail);
    }
  }, [currentUserEmail]);

  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setStatusMsg(null);

    const normalizedEmail = emailInput.trim().toLowerCase();
    const cleanPasskey = passkeyInput.trim();

    if (!normalizedEmail || !normalizedEmail.includes('@')) {
      setErrorMsg('Please enter a valid email address.');
      return;
    }
    if (cleanPasskey.length < 4) {
      setErrorMsg('Please enter a passkey (at least 4 characters).');
      return;
    }

    setSubmitting(true);
    try {
      const hashed = await hashPasskey(normalizedEmail, cleanPasskey);
      const users = loadUsers();
      const existing = users[normalizedEmail];

      if (!existing) {
        const credentialId = await tryRegisterWebAuthnPasskey(normalizedEmail);
        users[normalizedEmail] = {
          email: normalizedEmail,
          passkeyHash: hashed,
          credentialId,
        };
        saveUsers(users);
        onUserChange(normalizedEmail);
        setPasskeyInput('');
        setStatusMsg(`Registered passkey for ${normalizedEmail}.`);
      } else if (existing.passkeyHash === hashed) {
        onUserChange(normalizedEmail);
        setPasskeyInput('');
        setStatusMsg(`Signed in as ${normalizedEmail}.`);
      } else {
        setErrorMsg('Passkey does not match this email.');
      }
    } catch {
      setErrorMsg('Unable to verify passkey.');
    } finally {
      setSubmitting(false);
    }
  };

  const userHistory = currentUserEmail ? historyByUser[currentUserEmail] ?? [] : [];

  return (
    <div className="bg-[#FFFEFC] border border-[#E3D9CA] rounded-2xl p-6 space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#EAE2D6] pb-4">
        <div className="flex items-center gap-2.5">
          <div className="p-2 rounded-lg bg-[#FAF6F0] border border-[#DFD4C3] text-[#7C5333]">
            <KeyRound className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-[#231D18]">
              Passkey Account &amp; Search History
            </h2>
            <p className="text-xs text-[#6E6356]">
              Sign in with email and passkey to save and revisit your target pair searches
            </p>
          </div>
        </div>

        {currentUserEmail && (
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F2F8F4] border border-[#CBE0D2] text-xs font-medium text-[#2F5A3B]">
              <UserCheck className="w-3.5 h-3.5" />
              {currentUserEmail}
            </span>
            <button
              type="button"
              onClick={() => {
                onUserChange(null);
                setStatusMsg(null);
                setErrorMsg(null);
              }}
              className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg border border-[#DFD4C3] bg-[#FAF6F0] hover:bg-[#EFE7DC] text-xs font-medium text-[#574D43] cursor-pointer"
            >
              <LogOut className="w-3.5 h-3.5" />
              Sign out
            </button>
          </div>
        )}
      </div>

      {!currentUserEmail ? (
        <form onSubmit={handleAuthSubmit} className="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
          <div>
            <label htmlFor="auth-email" className="block text-xs font-medium text-[#574D43] mb-1">
              Email
            </label>
            <input
              id="auth-email"
              type="email"
              value={emailInput}
              onChange={(e) => setEmailInput(e.target.value)}
              placeholder="researcher@lab.org"
              className="w-full h-10 px-3 bg-[#FAF6F0] border border-[#DFD4C3] rounded-lg text-sm text-[#231D18] focus:outline-none focus:border-[#7C5333]"
            />
          </div>

          <div>
            <label htmlFor="auth-passkey" className="block text-xs font-medium text-[#574D43] mb-1">
              Passkey
            </label>
            <input
              id="auth-passkey"
              type="password"
              value={passkeyInput}
              onChange={(e) => setPasskeyInput(e.target.value)}
              placeholder="Enter passkey (4+ chars)"
              className="w-full h-10 px-3 bg-[#FAF6F0] border border-[#DFD4C3] rounded-lg text-sm text-[#231D18] focus:outline-none focus:border-[#7C5333]"
            />
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="h-10 px-4 bg-[#2D241E] hover:bg-[#1E1713] text-[#FFFEFC] text-xs font-medium rounded-lg transition-colors cursor-pointer disabled:opacity-50"
          >
            {submitting ? 'Verifying...' : 'Continue with Passkey'}
          </button>
        </form>
      ) : (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#231D18]">
              <History className="w-3.5 h-3.5 text-[#7C5333]" />
              <span>Your Search History ({userHistory.length})</span>
            </div>
            {userHistory.length > 0 && (
              <button
                type="button"
                onClick={() => onClearHistory(currentUserEmail)}
                className="inline-flex items-center gap-1 text-xs text-[#853826] hover:underline cursor-pointer"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Clear history
              </button>
            )}
          </div>

          {userHistory.length === 0 ? (
            <p className="text-xs text-[#7A6E61] bg-[#FAF6F0] border border-[#EAE2D6] rounded-xl p-3.5">
              No searches recorded for {currentUserEmail} yet. Evaluate a gene pair above to save it here.
            </p>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
              {userHistory.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => onSelectHistoryItem(item.gene1, item.gene2)}
                  className="text-left p-3 rounded-xl bg-[#FAF6F0] hover:bg-[#F3ECE1] border border-[#E3D9CA] transition-colors cursor-pointer"
                >
                  <div className="font-mono text-xs font-semibold text-[#231D18]">
                    {item.gene1} + {item.gene2}
                  </div>
                  <div className="mt-1 flex items-center justify-between text-[11px] text-[#6E6356]">
                    <span>
                      {typeof item.score === 'number'
                        ? `Score ${item.score.toFixed(4)}`
                        : 'Saved pair'}
                      {typeof item.rank === 'number' ? ` · #${item.rank}` : ''}
                    </span>
                    <span>{new Date(item.searchedAt).toLocaleTimeString()}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {statusMsg && (
        <div className="text-xs text-[#2F5A3B] bg-[#F2F8F4] border border-[#CBE0D2] rounded-lg px-3 py-2">
          {statusMsg}
        </div>
      )}
      {errorMsg && (
        <div className="text-xs text-[#853826] bg-[#FDF4F0] border border-[#E5C1B5] rounded-lg px-3 py-2">
          {errorMsg}
        </div>
      )}
    </div>
  );
}