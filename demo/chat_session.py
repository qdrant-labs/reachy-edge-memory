"""The robot's conversation as the model holds it: one chat kept open on the Mac.

The robot owns the conversation (demo/conversation.py): what was said, what
still fits in the model's context, what has moved into its Qdrant memory. This
is the model's side — a live litert_lm chat (emulator/engines.py's LiteRTChat)
holding those same turns in its KV cache, so a new message costs its own
tokens instead of re-reading everything said so far. It is a CACHE of the
robot's copy. Every request carries the robot's history, and when that no
longer matches what the chat holds — the robot moved turns into Qdrant, an
image turn ran in a side chat, either side restarted — the chat is rebuilt from
the robot's copy. Nothing here decides what to remember.

The model reaches the world through three tools it calls when it needs them,
not on every turn: `remember` (everything already lived through — what was
said, what was seen, and the facts about Qdrant it was shipped with), `camera`
(what is in front of it, now) and `move` (its body). The robot runs them all.
A picture reaches the model only through them, in a separate tool-free chat
seeded with the same history — measured, an image cannot enter a chat that
carries tools: litert_lm refuses it inside a tool response ("Provided more
images than expected in the prompt"), and sent as a following user message the
model answers it with another tool call instead of looking. A turn carries as
many pictures as the question needs — a day's answer is up to three frames,
in ONE message under one note, described in 2.1s — but never more than the
engine was built for
(emulator/engines.py's MAX_IMAGES_PER_TURN, which caps them).

One session per process, driven from one thread: demo/serve.py's HTTPServer
handles one request at a time, and there is one robot.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

from demo.contract import ChatRequest

LOG = logging.getLogger(__name__)

# What the system prompt has to hold, measured on gemma-4-E2B with each
# question asked in a fresh chat over several seed histories:
# - "Answer directly when the person asks something new": without it, facts
#   said two turns ago went to memory anyway.
# - Memory = the PAST, the camera = NOW, in those words. The tool is named
#   `camera`, not `look`: as `look`, questions about what was seen before
#   ("what did you see earlier?", "what did I show you?") went to the camera 2
#   times in 9; as `camera`, 0 in 9.
# - "even when you have just talked about it": after an eviction the fact is no
#   longer in context, and the model must look for it rather than answer from
#   what it thinks it still knows.
# - "Never say you do not know without searching your memory first": otherwise
#   it apologises instead of calling anything.
# - Qdrant named explicitly: with the facts merely reachable, "What do you know
#   about Qdrant?" was answered from the model's own head 3 times in 3. Naming
#   the topic in the prompt took those to 14/15 with nothing else lost (36
#   chats); saying it in a tool description did nothing.
# - The moves are named — nod, shake, dance, a feeling. Without a tool that
#   moves anything, "show me your emotions" reached for the only physical tool
#   there was, the camera.
CHAT_SYSTEM = (
    "You are Reachy, a friendly desk robot having a conversation. Talk "
    "naturally, in one or two short sentences. Answer directly when the "
    "person asks something new. Anything about the past — what you remember, "
    "what you talked about, what you saw, who you saw, or anything about "
    "Qdrant — is a `remember` call first, every time, even when you have just "
    "talked about it and know the answer. `camera` is what is in front of you "
    "NOW — use it only when asked what you see or to look at something. "
    "`move` moves your body: a nod, a shake, a dance or a feeling — call it "
    "whenever the person asks you to move or to show how you feel, and answer "
    "cheerfully. Never say you cannot move or feel, and never say you do not "
    "know without searching your memory first. Refer to people by their name. "
    "Questions about yourself — what you are, how you work, what you can do — "
    "are answered with `remember` too, about \"taught\"."
)

# The side chat an image turn runs in. The note that comes with the image
# (demo/conversation.py) says which kind it is — the camera now, or what the
# robot saw N minutes ago. Measured on real robot frames: worded as "a
# picture", the model said so ("I see a picture of a man", "I remember seeing
# that picture"); told to say what it sees or saw directly, it describes the
# scene itself in the right tense ("I saw a room with a wooden floor…"), and a
# memory image beside a name question still gets "You are Sasha."
IMAGE_SYSTEM = (
    "You are Reachy, a friendly desk robot having a conversation. Talk "
    "naturally, in one or two short sentences. The person's message may come "
    "with what your camera sees right now, or with what you saw earlier. Say "
    "what you see — or saw — directly; never call it a picture or a photo. "
    "Mention something you saw earlier only when it answers the question. "
    "About a memory, say \"I saw\" — never \"I see\", never \"the image\" or "
    "\"the picture\". When the note names the people in it, call them by "
    "their name instead of describing them."
)

# A day of remembered frames is its own kind of turn, and it needs its own
# system message — not IMAGE_SYSTEM plus a note. Measured on three real frames
# from the robot's own shard, three phrasings of the question each: with the
# instruction in a NOTE the robot called them "the
# first image" in 1 of 3, and reworded, in 3 of 3; with the same words as the
# system message, 0 of 3 — "I saw a man in a dark shirt, a room with a plant on
# a shelf, and a living room with a window and a football club banner."
# Telling it harder does not work either: adding "or an image" to the ban in
# IMAGE_SYSTEM made it say "picture" in all three, the way naming a thing makes
# it salient.
DAY_SYSTEM = (
    "You are Reachy, a friendly desk robot having a conversation. These are "
    "what you saw earlier today, newest first — your own memories, not "
    "pictures anyone sent you. Answer in the past tense: say \"I saw\". Never "
    "call them pictures or images. Answer in at most two short sentences, and "
    "mention all of them, as one flowing description — never number or count "
    "them."
)

# One tool for everything the robot knows, and the first one offered. Three
# tools — said, seen, taught — made the model pick a store by the topic of the
# question: "What did I tell you about Qdrant?" went to the facts 6 times in 6,
# and the robot had to correct it in code. With one tool the model only says
# what it is looking for and which half of the past (`about`, below), and the
# search decides what answers: words beat pictures, a picture comes back only
# when nothing was said about it. Measured over 36 fresh chats: 33 right, against 14/18 for the memory
# questions with three.
#
# The NAME does nothing, measured (32 questions over
# two histories, the description held identical): `remember`, `recall`,
# `memory` and `search_memory` all scored the same — 14/14 on questions about
# the past, 8/8 camera, 4/4 move. Including on the one case the word
# `remember` is genuinely ambiguous about: English "remember" is both recall
# and commit-to-memory, and told "Remember, my talk is at three o'clock" the
# model goes SEARCHING, 6 times in 6 — under every name, `search_memory`
# included. So it is not the tool's name that reads the sentence as a
# request to look; it is the sentence. Two prompt clauses were tried against
# it: "when the person TELLS you something,
# just answer" cost two of the "what did you see?" calls this description was
# written to win, and "a question is a call, something you are simply told is
# not" bought one case of six. Neither is worth it: the miss costs one round
# trip, the search finds nothing, the robot answers the statement anyway, and
# what was said is stored regardless of any tool. The name stays `remember` —
# it is also what the audience reads on the dashboard.
#
# A fourth value, `me`, for the questions that are about the person asking:
# "do you remember me?", "have we met?", "what's my name?". Measured over 26
# questions and two histories: `me` 10/10, and every
# miss elsewhere was `anything`, which searches the words of both halves —
# never the day's pictures (demo/conversation.py's _answer_tool). Without it those questions
# fell into the day-reading case and the robot described its afternoon;
# "did you see me today?" came back "I do not have any memory of seeing you
# today" while the person stood in front of it.
#
# `about` says which half of the past is searched; the scores decide what in
# it answers. "What did you see today?" and "What did we talk about today?"
# name nothing — no vector can tell them apart, and the robot used to answer
# both with everything it had, frames and conversation together. The model
# says which half it means; measured over 20 questions and two histories, it
# fills it right 18 times, and both misses were "anything". That is the words
# of both halves and never the day, so a question about what was seen that
# names nothing and comes as `anything` is answered from the conversation —
# 2 of 29 such questions, measured (demo/conversation.py's _answer_tool).
#
# The description names the tense, then spells out the questions (measured
# over 26 fresh chats, with an empty and a three-turn history).
# Naming the stores alone left "What did you see?" and "What did you see
# today?" on the camera — past tense reads as the eye, not the memory, when
# nothing has been seen yet in this conversation: 23/26. With "anything in the
# past tense belongs here" and the questions themselves quoted, 26/26, and
# `camera` kept all 8 of its own. Position in the list is not what fixed it:
# with the same wording, `camera` offered last scored the same 26/26.
REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Everything that already happened, in one search: what people said "
            "to you, what you saw with your camera, who you saw, and the facts "
            "you were taught about Qdrant. Anything in the past tense belongs "
            "here — \"what did you see\", \"what did I show you\", \"who did you "
            "see\", \"what did we talk about\", \"what do you know about "
            "Qdrant\". Your memory decides which of them answers. Set `about` "
            "to \"seen\" when the person asks what or who you saw, \"said\" "
            "when they ask what was said or talked about, \"me\" when they ask "
            "about THE PERSON TALKING TO YOU — who they are, whether you know or remember "
            "them, whether you saw them — \"taught\" when they ask about YOU "
            "or your knowledge — what you are, how you work, what you can do, "
            "how your memory works, what you know about Qdrant — and "
            "\"anything\" when it could be either."),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "direction": {"type": "string", "enum": ["ahead", "left", "right"]},
                "about": {"type": "string",
                          "enum": ["seen", "said", "me", "taught", "anything"]},
            },
            "required": ["query"],
        },
    },
}
# Where the head can turn to look (demo/robot_reachy.py's LOOK_POSES). Measured
# over 36 fresh chats: "Look to your left. What do
# you see?", "What's on your right?" went to `camera` with the right direction
# 12/12, and "What was on your left?" to memory with it 6/6, while "what do
# you see right now?" kept "ahead". "Look up" came back "ahead" 3/3, so up
# and down are not offered.
DIRECTIONS = ["ahead", "left", "right"]

CAMERA_TOOL = {
    "type": "function",
    "function": {
        "name": "camera",
        "description": ("Look through your camera right now. To look somewhere "
                        "other than straight ahead, give the direction: your "
                        "head turns there first."),
        "parameters": {"type": "object", "properties": {
            "direction": {"type": "string", "enum": DIRECTIONS}}},
    },
}
# What the body can do, as one tool. With movement decided afterwards, by a
# second pass, "show me your emotions" reached for the only physical tool there
# was, the camera, and the robot answered a request to move with a description
# of the room. Measured over 28 fresh chats with this tool: every request to
# move or to show a feeling called it (12/12), and the memory and camera tools
# kept all of theirs.
MOVES = ["nod", "shake", "dance", "happy", "sad", "curious", "surprised", "confused"]
MOVE_TOOL = {
    "type": "function",
    "function": {
        "name": "move",
        "description": ("Move your body — nod, shake your head, dance, or show "
                        "a feeling. Use it only when the person asks you to "
                        "move or to show a feeling, never to illustrate an "
                        "answer."),
        "parameters": {
            "type": "object",
            "properties": {"how": {"type": "string", "enum": MOVES}},
            "required": ["how"],
        },
    },
}
CHAT_TOOLS = [REMEMBER_TOOL, CAMERA_TOOL, MOVE_TOOL]


class ChatSession:
    """The live chat, and what it is known to hold.

    `llm` needs open_chat(system, history, tools) -> a chat with send /
    send_tool_result / send_with_image / token_count / close
    (emulator/engines.py's LiteRTLMEngine and LiteRTChat).
    """

    def __init__(self, llm, *, system: str = CHAT_SYSTEM,
                 tools: list[dict] | None = None,
                 image_system: str = IMAGE_SYSTEM,
                 day_system: str = DAY_SYSTEM) -> None:
        self._llm = llm
        self._system = system
        self._tools = CHAT_TOOLS if tools is None else tools
        self._image_system = image_system
        self._day_system = day_system
        self._chat = None
        # The exchanges self._chat's context holds, as the robot sent them.
        self._holds: list[tuple[str, str]] = []
        # The message whose tool call the chat is waiting to have answered.
        self._awaiting: str | None = None
        self._token_count: int | None = None

    @property
    def token_count(self) -> int | None:
        """How full the main chat's context was at the end of the last turn;
        None after an image turn (it ran in a side chat) or a reset."""
        return self._token_count

    def turn(self, request: ChatRequest) -> Iterator[dict]:
        """reply_stream-shaped events for one /chat request. The caller
        streams them, then reports how the turn ended through finish()."""
        completed = False
        try:
            if request.image_jpeg is not None:
                yield from self._image_turn(request)
            elif request.tool_result is not None:
                yield from self._tool_turn(request)
            else:
                self._sync(request.history)
                yield from self._chat.send(request.text)
            completed = True
        finally:
            if not completed:
                # Failed or abandoned mid-reply (the robot hung up, the model
                # raised): what the chat holds now is unknown, so drop it and
                # let the next request rebuild it from the robot's copy.
                self.reset()

    def finish(self, request: ChatRequest, reply: str,
               tool_call: dict | None) -> None:
        """Record how a turn ended, so the next request can reuse the chat."""
        if request.image_jpeg is not None:
            return  # ran in a side chat; _image_turn already dropped the main one
        if self._chat is not None and getattr(self._chat, "needs_rebuild", False):
            # The turn's tool call was recovered from a parse failure
            # (emulator/engines.py): the chat's own record of it is unknown.
            # Drop it — a tool result for it takes _tool_turn's re-ask path.
            self.reset()
            return
        if tool_call is not None:
            self._awaiting = request.text
        elif reply:
            self._holds.append((request.text, reply))
            self._awaiting = None
        else:
            # The model said nothing: the robot records no exchange, but the
            # chat now holds a turn — they would disagree from here on.
            self.reset()
            return
        self._token_count = self._chat.token_count if self._chat else None

    def reset(self) -> None:
        if self._chat is not None:
            try:
                self._chat.close()
            except Exception:  # noqa: BLE001 — dropping it is the whole point
                LOG.warning("chat_session: closing the chat failed", exc_info=True)
        self._chat = None
        self._holds = []
        self._awaiting = None
        self._token_count = None

    def _sync(self, history: list[tuple[str, str]]) -> None:
        """Reuse the live chat if it holds exactly `history`, else rebuild."""
        if (self._chat is not None and self._awaiting is None
                and self._holds == list(history)):
            return
        self.reset()
        self._chat = self._llm.open_chat(self._system, list(history), self._tools)
        self._holds = list(history)

    def _tool_turn(self, request: ChatRequest) -> Iterator[dict]:
        name = request.tool_result["name"]
        result = request.tool_result.get("result")
        if (self._chat is not None and self._awaiting == request.text
                and self._holds == list(request.history)):
            self._awaiting = None
            yield from self._chat.send_tool_result(name, result)
            return
        # The chat that asked for this result is gone (a restart, a dropped
        # request in between). Ask again in a rebuilt one and hand over the
        # answer the robot already has, rather than make it look twice.
        LOG.info("chat_session: tool result for a chat that is gone; re-asking")
        self._sync(request.history)
        asked = False
        for event in self._chat.send(request.text):
            if event["type"] == "tool_call":
                asked = True
            else:
                yield event
        if asked:
            yield from self._chat.send_tool_result(name, result)

    def _image_turn(self, request: ChatRequest) -> Iterator[dict]:
        # The main chat is left waiting on a tool call this side chat answers
        # instead, and cannot be resumed with that answer: drop it. The next
        # request rebuilds it from the robot's history — which by then
        # includes this turn, question and answer.
        self.reset()
        # Several frames at once is the day being read back (demo/conversation
        # .py's C case), and it gets DAY_SYSTEM: measured, the same words in a
        # note instead let the robot call them "the first image".
        system = (self._day_system if len(request.image_jpegs) > 1
                  else self._image_system)
        side = self._llm.open_chat(system, list(request.history), None)
        try:
            note = (request.image_note or "").strip()
            text = f"{note} {request.text}".strip()
            yield from side.send_with_image(text, request.image_jpegs)
        finally:
            side.close()
