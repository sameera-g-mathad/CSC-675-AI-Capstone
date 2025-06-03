import { Stack } from "expo-router";
import { ChatHistContextProvider } from "@/app/_context/ChathistContext";
export default function layout() {
    return <ChatHistContextProvider>
        <Stack >

            <Stack.Screen
                name='[param]'
                options={
                    {
                        headerShown: true,
                        headerTitle: '',
                    }} />
        </Stack>
    </ChatHistContextProvider>
}