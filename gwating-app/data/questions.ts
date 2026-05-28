import { TraitKey } from "@/types/matching";

export type QuizChoice = {
  text: string;
  score: number;
};

export type QuizQuestion = {
  id: number;
  trait: TraitKey;
  situation: string;
  choices: QuizChoice[];
};

export const questions: QuizQuestion[] = [
  {
    id: 1,
    trait: "atmosphereCoordination",
    situation: "과팅 자리에서 대화가 갑자기 끊겼어요. 어떻게 할까요?",
    choices: [
      { text: "바로 새 주제를 꺼낸다", score: 5 },
      { text: "조금 기다리다가 자연스럽게 말을 건다", score: 4 },
      { text: "누군가 먼저 말하길 기다린다", score: 2 },
      { text: "조용해져도 크게 신경 쓰지 않는다", score: 1 },
    ],
  },
  {
    id: 2,
    trait: "atmosphereCoordination",
    situation: "분위기가 예상보다 많이 가라앉았어요.",
    choices: [
      { text: "가벼운 게임이나 공통 주제를 제안한다", score: 5 },
      { text: "옆 사람에게 먼저 말을 걸어본다", score: 4 },
      { text: "분위기가 바뀔 때까지 기다린다", score: 2 },
      { text: "그냥 빨리 끝나길 바란다", score: 1 },
    ],
  },
  {
    id: 3,
    trait: "consideration",
    situation: "말수가 거의 없는 사람이 한 명 있어요.",
    choices: [
      { text: "자연스럽게 그 사람에게 질문을 돌린다", score: 5 },
      { text: "눈을 맞추며 참여할 타이밍을 만든다", score: 4 },
      { text: "본인이 말하고 싶으면 하겠지라고 생각한다", score: 2 },
      { text: "크게 신경 쓰지 않는다", score: 1 },
    ],
  },
  {
    id: 4,
    trait: "consideration",
    situation: "상대가 불편해 보이는 상황이에요.",
    choices: [
      { text: "조용히 괜찮은지 물어본다", score: 5 },
      { text: "주제를 바꿔 분위기를 돌린다", score: 4 },
      { text: "일단 대화 흐름에 집중한다", score: 2 },
      { text: "못 본 척한다", score: 1 },
    ],
  },
  {
    id: 5,
    trait: "participation",
    situation: "친구가 게임을 제안했어요.",
    choices: [
      { text: "바로 좋다고 하며 참여한다", score: 5 },
      { text: "다수가 원하면 같이 한다", score: 4 },
      { text: "분위기를 보고 천천히 참여한다", score: 2 },
      { text: "별로지만 형식적으로만 참여한다", score: 1 },
    ],
  },
  {
    id: 6,
    trait: "participation",
    situation: "자기소개 순서가 돌아왔어요.",
    choices: [
      { text: "짧은 에피소드까지 더해 자연스럽게 말한다", score: 5 },
      { text: "준비한 말을 깔끔하게 한다", score: 4 },
      { text: "최대한 짧게 말한다", score: 2 },
      { text: "긴장해서 말이 잘 나오지 않는다", score: 1 },
    ],
  },
  {
    id: 7,
    trait: "respectfulness",
    situation: "상대가 대답하기 싫어하는 것 같은 질문을 받았어요.",
    choices: [
      { text: "불편하면 답하지 않아도 된다고 말한다", score: 5 },
      { text: "질문을 가볍게 바꿔서 이어간다", score: 4 },
      { text: "반응을 보고 괜찮으면 계속 묻는다", score: 2 },
      { text: "솔직한 게 좋으니 그대로 묻는다", score: 1 },
    ],
  },
  {
    id: 8,
    trait: "respectfulness",
    situation: "자리가 생각보다 가깝고 신체 접촉이 생길 것 같아요.",
    choices: [
      { text: "자연스럽게 거리를 만들거나 양해를 구한다", score: 5 },
      { text: "상대 표정을 보고 불편하면 조절한다", score: 4 },
      { text: "어색하지만 그냥 앉아 있는다", score: 2 },
      { text: "친해지는 과정이라고 생각한다", score: 1 },
    ],
  },
  {
    id: 9,
    trait: "communicationBalance",
    situation: "한 사람이 대화를 오래 독점하고 있어요.",
    choices: [
      { text: "다른 사람에게 질문을 넘겨 균형을 맞춘다", score: 5 },
      { text: "자연스럽게 끼어들어 주제를 바꾼다", score: 4 },
      { text: "나도 같이 참여하며 지켜본다", score: 2 },
      { text: "그 사람이 지칠 때까지 기다린다", score: 1 },
    ],
  },
  {
    id: 10,
    trait: "communicationBalance",
    situation: "지금 대화에서 내 파트너가 많이 조용해요.",
    choices: [
      { text: "가볍게 물어보며 대화에 들어올 틈을 만든다", score: 5 },
      { text: "관심사를 물어보며 편하게 이어간다", score: 4 },
      { text: "나도 조용히 있는다", score: 2 },
      { text: "불편해서 다른 대화에 끼려고 한다", score: 1 },
    ],
  },
];
