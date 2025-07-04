import React, { createContext, useCallback, useReducer, type PropsWithChildren, type ReactNode } from 'react'

// This context is used to manage the chat history and messages in the application.
// It provides functions to ask queries and fetch messages, along with the current state of the chat
// including messages, chat title, and display image.


type userQueryAction = {
    action: 'user_query',
    value: string
}

type updateResponseQuery = {
    action: 'update_response',
    value: [number, string]
}

type updateTitle = {
    action: 'update_title',
    value: string
}

type updateMessages = {
    action: 'update_messages',
    value: { displayImage: string, chat_title: string, messages: any }
}

type Message = {
    role: 'user' | 'bot',
    content: string
}

type ChatState = {
    messages: Message[];
    chat_title: string;
    displayImage: string
};

type ChatContextType = ChatState & {
    askQuery: (uuid: string, query: string) => Promise<void>;
    queryMessages: (uuid: string) => Promise<void>;
};

const ChatHistContext = createContext<ChatContextType>({
    messages: [],
    chat_title: '',
    displayImage: '',
    askQuery: async (uuid: string, query: string) => undefined,
    queryMessages: async (uuid: string) => undefined
})

const decoder = new TextDecoder();

/**
 * This function reads a stream of data from a ReadableStreamDefaultReader and yields the response.
 * It decodes the Uint8Array data into a string and parses it as JSON.
 * The function continues reading until the stream is finished.
 */
// It yields the response and indicates whether the stream is done.
async function* readStream(reader: ReadableStreamDefaultReader<Uint8Array<ArrayBufferLike>>) {
    let finished = false
    while (!finished) {
        let { value } = await reader.read();
        console.log(value)
        const { response, done } = JSON.parse(decoder.decode(value, { stream: true }))
        yield response;
        finished = done
    }
    return
}

// This reducer function manages the state of the chat history.
// It updates the state based on the action type and payload received.
// The actions include adding a user query, updating the bot's response, updating the chat title,
// and updating the messages with new data from the server.
// Each action modifies the state accordingly, ensuring that the chat history is maintained correctly.
// The state includes an array of messages, a chat title, and a display image.
// The messages are structured as an array of objects, each containing a role ('user' or 'bot') and content (the message text).
// The chat title is a string representing the title of the chat,
// and the display image is a string representing the URL or path to an image associated
const chathistReducer = (state: ChatState, payload: userQueryAction | updateResponseQuery | updateTitle | updateMessages): ChatState => {
    switch (payload.action) {
        case 'user_query': return {
            ...state,
            messages: [...state.messages, { role: 'user', content: payload.value }, { role: 'bot', content: '' }]

        };
        case 'update_response': return {
            ...state,
            messages: state.messages.map((el, index) => {
                if (index == payload.value[0]) {
                    return { role: 'bot', content: payload.value[1] }
                }
                return el;
            })
        }
        case 'update_title': return { ...state, chat_title: payload.value };
        case 'update_messages': {
            const displayImage = payload.value['displayImage'] !== undefined ? payload.value['displayImage'] : ''
            return { ...state, displayImage, chat_title: payload.value['chat_title'], messages: [...state.messages, ...payload.value['messages']] }
        }
        default: return state;
    }
}

// This context provider component wraps the application and provides the chat history context to its children.
// It initializes the state using a reducer and provides functions to query messages and ask questions.
// The `queryMessages` function fetches messages from the server based on a UUID and updates the state with the retrieved messages, chat title, and display image.
// The `askQuery` function sends a user query to the server, streams the response, and updates the state with the bot's response.
// If the chat title is still set to 'New Title', it also calls `getTitle` to generate a title based on the user query and bot response.
// The `getTitle` function sends a request to the server to generate a title based on the provided query and updates the state with the generated title.
// The context value includes the current state and the functions to interact with the chat history.
export const ChatHistContextProvider: React.FC<PropsWithChildren> = ({ children }) => {
    const [state, dispatch] = useReducer(chathistReducer, {
        messages: [],
        chat_title: 'New Title',
        displayImage: ''
    } as ChatState)

    // This function queries messages from the server based on a UUID.
    // It sends a POST request to the server with the UUID and retrieves the chat details and conversations.
    // If the chat details are undefined, it returns early.
    // Otherwise, it dispatches an action to update the state with the retrieved messages, chat title, and display image.
    const queryMessages = async (uuid: string) => {
        const response = await fetch('http://127.0.0.1:4000/api/v1/getMessages', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ uuid })
        })
        const json = await response.json();
        const chatDetails = json.data[0]
        const conversations = json.data['conversations']
        if (chatDetails === undefined) return
        dispatch({ action: 'update_messages', value: { displayImage: chatDetails['displayImage'], chat_title: chatDetails['chat_title'], messages: conversations } })
    }

    // This function sends a user query to the server and streams the response.
    // It dispatches an action to update the state with the user query.
    // It then sends a POST request to the server with the query and UUID.
    // If the response body is available, it reads the stream and updates the state with the bot's response as it is received.
    // If the chat title is still set to 'New Title', it calls the `getTitle` function to generate a title based on the user query and bot response.
    // Finally, it returns undefined.
    const askQuery = useCallback(async (uuid: string, query: string) => {
        let message = ''
        dispatch({ action: 'user_query', value: query });
        const response = await fetch('http://127.0.0.1:4000/api/v1/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query, uuid })
        })
        if (response.body) {
            const reader = response.body.getReader()
            for await (const response of readStream(reader)) {
                message += response
                dispatch({ action: 'update_response', value: [state.messages.length + 1, message] })
            }
        }
        else {
            console.log('Error while streaming.')
        }
        if (state.chat_title === 'New Title') {
            getTitle(uuid, query = `User: ${query}\n Bot: ${message}`)
        }
        return undefined;
    }, [state.messages.length])


    // This function generates a title for the chat based on the user query and bot response.
    // It sends a POST request to the server with the query and UUID.
    const getTitle = useCallback(async (uuid: string, query: string) => {
        let message = ''
        const response = await fetch('http://127.0.0.1:4000/api/v1/title', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt: query, uuid })
        })
        if (response.body) {
            const reader = response.body.getReader()
            for await (const response of readStream(reader)) {
                message += response
                dispatch({ action: 'update_title', value: message })
            }
        }
        else {
            console.log('Error while streaming.')
        }
        return undefined;
    }, [])
    return <ChatHistContext.Provider value={{ ...state, askQuery, queryMessages }}>
        {children}
    </ChatHistContext.Provider>
}

export default ChatHistContext;