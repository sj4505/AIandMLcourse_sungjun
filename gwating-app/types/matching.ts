export type TraitKey =
  | "atmosphereCoordination"
  | "consideration"
  | "participation"
  | "respectfulness"
  | "communicationBalance";

export type MoodKey =
  | "comfortableTalk"
  | "activeSocial"
  | "gamesAndDrinks"
  | "respectfulSafe"
  | "naturalIntro";

export type MemberRole =
  | "moodMaker"
  | "coordinator"
  | "considerate"
  | "reactor";

export type TraitScores = Record<TraitKey, number>;

export type TeamMember = {
  nickname: string;
  role: MemberRole;
  traits?: TraitScores;
  isLeader?: boolean;
};

export type UserProfile = {
  nickname: string;
  traits: TraitScores;
};

export type TeamProfile = {
  teamName: string;
  school: string;
  region: string;
  size: number;
  ageRange: string;
  mood: MoodKey;
  members: TeamMember[];
  availableTimes?: string[];
};

export type MatchResult = {
  team: TeamProfile;
  score: number;
  vibeScore: number;
  traitScore: number;
  roleScore: number;
  conditionScore: number;
  reasons: string[];
  label: "Strong vibe fit" | "Good with some differences" | "Different atmosphere preferences";
};
