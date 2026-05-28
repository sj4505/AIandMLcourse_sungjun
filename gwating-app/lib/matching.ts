import { MatchResult, MoodKey, TeamProfile } from "@/types/matching";
import { moodWeights } from "@/data/moodWeights";
import {
  buildRoleVector,
  clamp01,
  clampScore,
  roleBalanceScore,
  roleComplementarityScore,
  traitSimilarityScore,
} from "@/lib/scoring";

const MOOD_LABELS: Record<MoodKey, string> = {
  comfortableTalk: "편한 대화형",
  activeSocial: "활발한 친목형",
  gamesAndDrinks: "게임/술자리형",
  respectfulSafe: "예의/안전 중시형",
  naturalIntro: "자연스러운 소개팅형",
};

function calcVibeScore(myMood: MoodKey, theirMood: MoodKey): number {
  return clamp01(moodWeights[myMood]?.[theirMood] ?? 0.5);
}

function calcRoleScore(my: TeamProfile, their: TeamProfile): number {
  const myBalance = roleBalanceScore(my.members);
  const theirBalance = roleBalanceScore(their.members);
  const complementarity = roleComplementarityScore(
    buildRoleVector(my.members),
    buildRoleVector(their.members)
  );

  return clamp01(myBalance * 0.25 + theirBalance * 0.35 + complementarity * 0.4);
}

function parseRange(range: string): [number, number] | null {
  const match = range.match(/(\d{1,2})\s*~\s*(\d{1,2})/);
  if (!match) return null;

  const min = Number(match[1]);
  const max = Number(match[2]);
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  return [Math.min(min, max), Math.max(min, max)];
}

function calcAgeOverlapScore(myRange: string, theirRange: string): number {
  const my = parseRange(myRange);
  const their = parseRange(theirRange);
  if (!my || !their) return 0.5;

  const overlap = Math.max(0, Math.min(my[1], their[1]) - Math.max(my[0], their[0]) + 1);
  const span = Math.max(my[1], their[1]) - Math.min(my[0], their[0]) + 1;
  return clamp01(span > 0 ? overlap / span : 0);
}

function calcTimeScore(myTimes?: string[], theirTimes?: string[]): number {
  if (!myTimes?.length || !theirTimes?.length) return 0.5;
  const theirSet = new Set(theirTimes);
  const overlap = myTimes.filter((time) => theirSet.has(time)).length;
  return clamp01(overlap / Math.max(myTimes.length, theirTimes.length));
}

function calcConditionScore(my: TeamProfile, their: TeamProfile): number {
  const sizeDiff = Math.abs((my.size || my.members.length) - (their.size || their.members.length));
  const sizeScore = sizeDiff === 0 ? 1 : sizeDiff === 1 ? 0.5 : 0;
  const ageScore = calcAgeOverlapScore(my.ageRange, their.ageRange);
  const timeScore = calcTimeScore(my.availableTimes, their.availableTimes);

  return clamp01(sizeScore * 0.4 + ageScore * 0.3 + timeScore * 0.3);
}

function scoreToLabel(score: number): MatchResult["label"] {
  if (score >= 80) return "Strong vibe fit";
  if (score >= 60) return "Good with some differences";
  return "Different atmosphere preferences";
}

function generateReasons(
  my: TeamProfile,
  their: TeamProfile,
  vibeRaw: number,
  traitRaw: number,
  roleRaw: number,
  condRaw: number
): string[] {
  const reasons: string[] = [];

  if (vibeRaw >= 0.75) {
    reasons.push(`${MOOD_LABELS[my.mood]} 분위기에서 자연스럽게 이어질 가능성이 높아요.`);
  } else {
    reasons.push(`${MOOD_LABELS[their.mood]} 성향이 더해져 새로운 대화 흐름이 예상돼요.`);
  }

  if (traitRaw >= 0.75) {
    reasons.push("팀 성향 평균이 비슷해서 대화 속도와 배려 방식에서 잘 맞을 수 있어요.");
  } else if (traitRaw >= 0.5) {
    reasons.push("서로 다른 성향이 있어 역할을 나누면 균형 잡힌 분위기가 예상돼요.");
  }

  if (roleRaw >= 0.7) {
    reasons.push("분위기 메이커와 조율자 역할이 섞여 만남 진행에서 잘 맞을 수 있어요.");
  } else if (roleRaw < 0.45) {
    reasons.push("역할 구성이 비슷해서 초반 진행 방식은 미리 맞춰보면 좋아요.");
  }

  if (condRaw >= 0.75) {
    reasons.push("인원과 나이대 조건이 가까워 만남을 잡기 쉬울 가능성이 높아요.");
  } else if (reasons.length < 3) {
    reasons.push("조건 일부는 다르지만 분위기 조율로 편한 만남이 예상돼요.");
  }

  while (reasons.length < 2) {
    reasons.push("첫 대화 주제만 가볍게 맞추면 무리 없는 분위기가 예상돼요.");
  }

  return reasons.slice(0, 3);
}

export function calculateMatchScore(
  myTeam: TeamProfile,
  candidate: TeamProfile
): MatchResult {
  const vibeRaw = calcVibeScore(myTeam.mood, candidate.mood);
  const traitRaw = traitSimilarityScore(myTeam, candidate);
  const roleRaw = calcRoleScore(myTeam, candidate);
  const condRaw = calcConditionScore(myTeam, candidate);
  const score = clampScore(vibeRaw * 40 + traitRaw * 35 + condRaw * 25);

  return {
    team: candidate,
    score,
    vibeScore: clampScore(vibeRaw * 100),
    traitScore: clampScore(traitRaw * 100),
    roleScore: clampScore(roleRaw * 100),
    conditionScore: clampScore(condRaw * 100),
    reasons: generateReasons(myTeam, candidate, vibeRaw, traitRaw, roleRaw, condRaw),
    label: scoreToLabel(score),
  };
}

export function rankTeams(
  myTeam: TeamProfile,
  candidates: TeamProfile[]
): MatchResult[] {
  return candidates
    .map((candidate) => calculateMatchScore(myTeam, candidate))
    .sort((a, b) => b.score - a.score);
}
