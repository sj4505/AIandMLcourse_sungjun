import { TraitKey, MemberRole, TeamMember, TeamProfile, TraitScores } from "@/types/matching";

export const TRAIT_KEYS: TraitKey[] = [
  "atmosphereCoordination",
  "consideration",
  "participation",
  "respectfulness",
  "communicationBalance",
];

export const MEMBER_ROLES: MemberRole[] = [
  "moodMaker",
  "coordinator",
  "considerate",
  "reactor",
];

const ROLE_WEIGHTS: Record<MemberRole, TraitScores> = {
  moodMaker: {
    atmosphereCoordination: 0.6,
    consideration: 0,
    participation: 0.4,
    respectfulness: 0,
    communicationBalance: 0,
  },
  coordinator: {
    atmosphereCoordination: 0.4,
    consideration: 0,
    participation: 0,
    respectfulness: 0,
    communicationBalance: 0.6,
  },
  considerate: {
    atmosphereCoordination: 0,
    consideration: 0.6,
    participation: 0,
    respectfulness: 0.4,
    communicationBalance: 0,
  },
  reactor: {
    atmosphereCoordination: 0,
    consideration: 0,
    participation: 0.5,
    respectfulness: 0,
    communicationBalance: 0.5,
  },
};

const ROLE_TRAIT_DEFAULTS: Record<MemberRole, TraitScores> = {
  moodMaker: {
    atmosphereCoordination: 5,
    consideration: 3,
    participation: 5,
    respectfulness: 3,
    communicationBalance: 3,
  },
  coordinator: {
    atmosphereCoordination: 4,
    consideration: 3,
    participation: 3,
    respectfulness: 4,
    communicationBalance: 5,
  },
  considerate: {
    atmosphereCoordination: 3,
    consideration: 5,
    participation: 3,
    respectfulness: 5,
    communicationBalance: 3,
  },
  reactor: {
    atmosphereCoordination: 3,
    consideration: 3,
    participation: 4,
    respectfulness: 3,
    communicationBalance: 5,
  },
};

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  return Math.min(max, Math.max(min, value));
}

export function clamp01(value: number): number {
  return clamp(value, 0, 1);
}

export function clampScore(value: number): number {
  return Math.round(clamp(value, 0, 100));
}

export function normalizeTraitValue(value: unknown): number {
  return clamp(typeof value === "number" ? value : 3, 1, 5);
}

export function normalizeTraits(traits?: Partial<Record<TraitKey, number>>): TraitScores {
  return TRAIT_KEYS.reduce((acc, key) => {
    acc[key] = normalizeTraitValue(traits?.[key]);
    return acc;
  }, {} as TraitScores);
}

export function classifyRole(traits: Record<TraitKey, number>): MemberRole {
  const safeTraits = normalizeTraits(traits);
  let best: MemberRole = "coordinator";
  let bestScore = -1;

  for (const role of MEMBER_ROLES) {
    const weights = ROLE_WEIGHTS[role];
    const score = TRAIT_KEYS.reduce(
      (sum, key) => sum + safeTraits[key] * weights[key],
      0
    );
    if (score > bestScore) {
      bestScore = score;
      best = role;
    }
  }
  return best;
}

export type RoleVector = Record<MemberRole, number>;

export function buildRoleVector(members: Pick<TeamMember, "role">[] = []): RoleVector {
  const counts: RoleVector = { moodMaker: 0, coordinator: 0, considerate: 0, reactor: 0 };
  for (const member of members) {
    if (MEMBER_ROLES.includes(member.role)) counts[member.role]++;
  }

  const total = Math.max(1, Object.values(counts).reduce((sum, count) => sum + count, 0));
  return {
    moodMaker: counts.moodMaker / total,
    coordinator: counts.coordinator / total,
    considerate: counts.considerate / total,
    reactor: counts.reactor / total,
  };
}

export function roleBalanceScore(members: Pick<TeamMember, "role">[] = []): number {
  const vector = buildRoleVector(members);
  const presentRoles = MEMBER_ROLES.filter((role) => vector[role] > 0).length;
  const diversityScore = presentRoles / MEMBER_ROLES.length;
  const concentrationPenalty = Math.max(...MEMBER_ROLES.map((role) => vector[role]));

  return clamp01(diversityScore * 0.75 + (1 - concentrationPenalty) * 0.25);
}

export function roleComplementarityScore(a: RoleVector, b: RoleVector): number {
  const overlap = MEMBER_ROLES.reduce((sum, role) => sum + Math.min(a[role], b[role]), 0);
  const combinedBalance = 1 - Math.abs(roleBalanceFromVector(a) - roleBalanceFromVector(b));
  return clamp01((1 - overlap) * 0.7 + combinedBalance * 0.3);
}

function roleBalanceFromVector(vector: RoleVector): number {
  const presentRoles = MEMBER_ROLES.filter((role) => vector[role] > 0).length;
  const concentrationPenalty = Math.max(...MEMBER_ROLES.map((role) => vector[role]));
  return clamp01((presentRoles / MEMBER_ROLES.length) * 0.75 + (1 - concentrationPenalty) * 0.25);
}

export function averageTeamTraits(team: TeamProfile): TraitScores {
  const members = Array.isArray(team.members) ? team.members : [];
  const traitRows = members.length > 0
    ? members.map((member) => normalizeTraits(member.traits ?? ROLE_TRAIT_DEFAULTS[member.role]))
    : [normalizeTraits()];

  return TRAIT_KEYS.reduce((acc, key) => {
    const sum = traitRows.reduce((total, traits) => total + traits[key], 0);
    acc[key] = normalizeTraitValue(sum / traitRows.length);
    return acc;
  }, {} as TraitScores);
}

export function traitSimilarityScore(myTeam: TeamProfile, theirTeam: TeamProfile): number {
  const myTraits = averageTeamTraits(myTeam);
  const theirTraits = averageTeamTraits(theirTeam);
  const averageDiff =
    TRAIT_KEYS.reduce((sum, key) => sum + Math.abs(myTraits[key] - theirTraits[key]), 0) /
    TRAIT_KEYS.length;

  return clamp01(1 - averageDiff / 4);
}
