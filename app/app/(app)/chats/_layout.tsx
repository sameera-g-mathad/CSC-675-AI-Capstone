import { Stack } from "expo-router";

export default function layout() {
    return <Stack>
        <Stack.Screen name='[param]' options={{ headerShown: true, contentStyle: { paddingBottom: 0 } }} />
    </Stack>
}