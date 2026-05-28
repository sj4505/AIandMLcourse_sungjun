import {
  MEMBER_ROLES,
  normalizeTraits,
} from "@/lib/scoring";
import { MoodKey, TeamMember, TeamProfile, UserProfile } from "@/types/matching";

const KEYS = {
  user: "gwating_user",
  team: "gwating_team",
} as const;

const MOODS: MoodKey[] = [
  "comfortableTalk",
  "activeSocial",
  "gamesAndDrinks",
  "respectfulSafe",
  "naturalIntro",
];

function getStorage(): Storage | null {
  if (typeof window !== "undefined") return window.localStorage;
  return typeof globalThis.localStorage === "undefined" ? null : globalThis.localStorage;
}

function readJson(key: string): unknown | null {
  const storage = getStorage();
  if (!storage) return null;

  try {
    const raw = storage.getItem(key);
    return raw ? JSON.parse(raw) : null;
  } catch {
    storage.removeItem(key);
    return null;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function normalizeNickname(value: unknown): string {
  return typeof value === "string" && value.trim() ? value.trim() : "익명";
}

function normalizeMember(value: unknown): TeamMember | null {
  if (!isRecord(value)) return null;
  const role = value.role;
  if (typeof role !== "string" || !MEMBER_ROLES.includes(role as TeamMember["role"])) return null;

  return {
    nickname: normalizeNickname(value.nickname),
    role: role as TeamMember["role"],
    isLeader: value.isLeader === true,
    traits: isRecord(value.traits) ? normalizeTraits(value.traits) : undefined,
  };
}

function normalizeStringArray(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const items = value.filter((item): item is string => typeof item === "string" && item.trim() !== "");
  return items.length > 0 ? items : undefined;
}

export function normalizeUser(value: unknown): UserProfile | null {
  if (!isRecord(value) || !isRecord(value.traits)) return null;

  return {
    nickname: normalizeNickname(value.nickname),
    traits: normalizeTraits(value.traits),
  };
}

export function normalizeTeam(value: unknown): TeamProfile | null {
  if (!isRecord(value)) return null;

  const mood = value.mood;
  if (typeof mood !== "string" || !MOODS.includes(mood as MoodKey)) return null;

  const members = Array.isArray(value.members)
    ? value.members.map(normalizeMember).filter((member): member is TeamMember => member !== null)
    : [];

  const ageRange = typeof value.ageRange === "string" && value.ageRange.trim()
    ? value.ageRange.trim()
    : "22~24";

  return {
    teamName: typeof value.teamName === "string" && value.teamName.trim() ? value.teamName.trim() : "이름 없는 팀",
    school: typeof value.school === "string" && value.school.trim() ? value.school.trim() : "부산대학교",
    region: typeof value.region === "string" && value.region.trim() ? value.region.trim() : "부산",
    size: Number.isFinite(value.size) ? Math.max(1, Math.round(Number(value.size))) : Math.max(1, members.length),
    ageRange,
    mood: mood as MoodKey,
    members,
    availableTimes: normalizeStringArray(value.availableTimes),
  };
}

export function saveUser(profile: UserProfile): void {
  getStorage()?.setItem(KEYS.user, JSON.stringify(profile));
}

export function loadUser(): UserProfile | null {
  return normalizeUser(readJson(KEYS.user));
}

export function saveTeam(team: TeamProfile): void {
  getStorage()?.setItem(KEYS.team, JSON.stringify(team));
}

export function loadTeam(): TeamProfile | null {
  return normalizeTeam(readJson(KEYS.team));
}

export function clearAll(): void {
  const storage = getStorage();
  storage?.removeItem(KEYS.user);
  storage?.removeItem(KEYS.team);
}
